# Scheduler Admission Diagnostics

This opt-in diagnostic logs scheduler CPU observations on the CANN v0.32
baseline. It covers the upstream scheduler and existing Ascend subclasses,
including ordinary P-side scheduling, async scheduling, and D-side recompute
scheduling. It is independent of Model Runner V1/V2 and does not require the
full `VLLM_ASCEND_TRACE` timeline.

## Enable

Set these variables **before starting each P or D service**:

```bash
export VLLM_ASCEND_SCHEDULER_ADMISSION_DEBUG=1
export VLLM_ASCEND_SCHEDULER_ADMISSION_LOG_INTERVAL_MS=1000
# Keep your existing vllm serve command and DP/TP configuration.
```

## Apply the paired vLLM core patch

Ordinary P-side scheduling delegates to `Scheduler.schedule()` in the separate
**vLLM core** package. Updating only vllm-ascend covers the Ascend overrides,
but does not add branch probes to that upstream method.

Stop the service and use the same Python environment as `vllm serve`:

```bash
# Run from the updated vllm-ascend repository, on both P and D nodes.
python3 tools/diagnostics/apply_scheduler_admission_core_patch.py \
    --vllm-root /vllm-workspace/vllm --check
python3 tools/diagnostics/apply_scheduler_admission_core_patch.py \
    --vllm-root /vllm-workspace/vllm --apply
```

`--vllm-root` accepts a repository root, package directory, or site-packages
root. Omit it to locate the installed top-level vLLM package without importing
vLLM or torch. The default action is a read-only check.

The bundled patch targets the v0.32 scheduler at core commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. The tool compares the schedule AST,
checks the patch, preserves an original-file backup, and verifies that only
that method changed. A different scheduling implementation is rejected without
editing it; provide that runtime core SHA/source to adapt the patch. Unrelated
changes elsewhere in the file are allowed when the method and patch contexts
are compatible. This edits Python source and requires a service restart, not
custom-kernel recompilation.

If the core hooks are missing, startup emits `core branch hooks are absent`.
Apply the paired patch before interpreting ordinary P-side branch reasons.
To restore the saved core file after stopping the service:

```bash
python3 tools/diagnostics/apply_scheduler_admission_core_patch.py \
    --vllm-root /vllm-workspace/vllm --restore
```

Restoration refuses to overwrite edits made after patch application. The backup
and its checksum record remain alongside `scheduler.py`.

## Log frequency

The interval is a positive integer in milliseconds, measured independently per
EngineCore. The default is 1000 ms. Each summary includes at most four helper event samples and four branch
samples; additional observations remain in the counters. Logs use INFO level.

Search the normal service log:

```bash
grep -F '[SchedulerAdmission]' serve.log
```

Disable by removing the debug variable (or setting it to `0`) and restarting.
When disabled, the scheduler diagnostic patch is not imported or installed.
Python source changes require a service restart; an editable installation does
not require recompiling custom kernels for this change.

## Read the log

Each line contains a JSON object with the following fields:

| Field | Meaning |
| --- | --- |
| `pid`, `dp_rank`, `scheduler` | EngineCore PID, DP rank, and actual scheduler class. |
| `timestamp` | Wall-clock seconds, useful for correlating with other logs. |
| `window_ms`, `steps` | Measurement window and number of completed `schedule()` calls. |
| `schedule_total_ms`, `schedule_max_ms` | Total and maximum scheduler wall time in this window. Includes diagnostic helper recording, but excludes snapshot collection and summary log output. |
| `latest` | Resource snapshot at the latest schedule return, rather than the window maximum. |
| `branch_reasons` | Counts from actual waiting-path stops, described below. |
| `branch_sources` | Schedule implementations that entered an instrumented waiting-admission region in this window. |
| `branch_samples` | Up to four actual branch events, with file/line location, action, remaining token budget, timestamp, and RequestID when one was already selected. |
| `omitted_branch_samples` | Branch events omitted from samples, still included in counters. |
| `observations` | Counts of actual helper outcomes. These are call/event counts, not unique denied requests. |
| `step_limits` | Counts of steps with backlog and an observed capacity or policy limit at return. These are not attributed to individual requests. |
| `helper_timings` | Call count, total and maximum duration of each observed helper. |
| `samples` | Up to four observed events with internal RequestID, status and waiting/running phase. |
| `omitted_samples` | Events omitted from the bounded sample list; still included in counters. |

### Actual waiting-path branches

The following calls are directly inside each supported `schedule()` body,
immediately before its actual `break` or `continue`. The log collector remains
shared and rate-limited; it no longer needs to guess these reasons from return
snapshots.

| Branch reason | Meaning |
| --- | --- |
| `active_request_limit` | The actual active-request-cap branch stopped admission. |
| `token_budget_exhausted` | The waiting loop naturally ended with no token budget; a different `break` does not trigger this event. |
| `input_budget_exhausted` | Remaining input budget cannot accommodate drafting slots. |
| `dp_balance_running_limit`, `dp_admission_filter` | The corresponding balance/admission policy branch blocked progress. |
| `request_state_not_ready`, `pending_output_tokens` | The selected request's state or outstanding output prevented admission. |
| `cache_lookup_pending`, `encoder_cache_pending` | Cache matching or encoder cache availability is unresolved. |
| `kv_allocation_failed` | KV allocation failed and the actual waiting-path stop was taken. |
| `remote_kv_load_in_progress` | Async KV loading was initiated; the request stayed out of RUNNING. |
| `lora_limit`, `prefill_cadence_gate`, `unchunked_prefill_budget` | The corresponding LoRA, prefill cadence or unchunked budget guard was taken. |
| `speculative_padding_budget`, `mamba_alignment_no_tokens`, `no_schedulable_tokens`, `recompute_budget_or_context_limit` | The associated shape/alignment/budget guard stopped this request. |
| `preemption_in_step`, `recompute_in_step`, `scheduler_paused` | The scheduler skipped its entire waiting-admission region for this step. |
| `no_eligible_waiting_queue` | No queue was eligible, despite remaining backlog; ordinary empty-queue completion is suppressed. |

These are per-step events, not permanent request rejections or unique request
counts. Stops before `peek_request()` have `scope=waiting_queue` and a null
RequestID; the logger does not peek or scan solely to invent an ID. Once a
request is selected, the sample has `scope=request` and its actual internal
RequestID. `location` is the actual runtime Python filename and line number.
The phase is `waiting`: running-request retry/skip branches are not exhaustively
instrumented.

`branch_sources` should show `vllm.v1.core.sched.scheduler.Scheduler.schedule`
for ordinary core scheduling, or the concrete Ascend override. A healthy step
can have sources and no blocking branch. Missing core coverage is also reported
by the startup warning above.

### Confirmed helper observations

| Observation | What actually happened |
| --- | --- |
| `cache_lookup_pending` | The root connector's `get_num_new_matched_tokens()` returned a pending match (`None` token count). |
| `kv_allocation_failed` | `KVCacheManager.allocate_slots()` returned `None`. Check the sample phase: a running request may retry after preemption and subsequently succeed. |
| `request_state_not_ready` | `_handle_blocked_waiting_request()` returned false. The sample retains the actual state, such as `WAITING_FOR_REMOTE_KVS`. |

`helper_timings` observes `allocate_slots`, `local_cache_lookup`
(`get_computed_blocks`), `cache_match`, and `blocked_state` when those helpers
exist. A connector lookup duration is **not** the complete KV transfer duration.

### Step-level constraints

| Constraint | Meaning at schedule return |
| --- | --- |
| `active_limit_reached` | Running plus streaming-input-waiting slots reached the active-request limit while backlog remained. |
| `token_budget_exhausted` | Scheduled tokens reached the configured step token budget while backlog remained. |
| `scheduler_paused` | The scheduler was not `UNPAUSED` and backlog remained. |
| `dp_balance_running_limit_reached` | Balance scheduling was enabled and its CPU running-count snapshot contained a DP at the running cap. |

Multiple constraints can coexist. They are evidence for the next investigation,
not proof of which branch prevented every waiting request from running. Use `branch_reasons` to identify actual executed waiting-path stops. The
return-time constraints alone cannot attribute a stop to an individual request.
An empty branch counter does not explain delays outside these waiting paths,
such as Worker preparation or running-prefill chunk intervals.

`latest` includes ordinary and KV-holding waiting counts, active and running
limits, token budget and scheduled tokens, new-request count, free KV blocks,
pause state, and optional DP balance counts. It does not query all cache groups
or scan the waiting queue. A free-block snapshot alone does not prove KV
allocation can succeed.

## Interpret a long delay

1. High request queue latency with short scheduler calls: investigate admission
   limits, workload, long in-flight steps, DP skew and cache dependencies.
2. High scheduler time: compare helper durations, then capture EngineCore CPU
   stacks and OS scheduling evidence. Wall time includes time not running on a
   CPU; it does not establish a CPU-compute bottleneck by itself.
3. Pending cache queries or blocked remote-KV states: correlate connector and
   transfer logs; this diagnostic does not measure the network transfer itself.
4. If the full timeline is also enabled, its per-request waiting-queue logging
   can add significant work. Run a control without full trace before attributing
   a delay to the scheduler.

Do not add helper times to scheduler time: the helper calls are inside the
scheduler interval. This interval excludes Worker input preparation, NPU model
execution and post-schedule DP collectives. It is also different from the full
engine-step duration and per-request queue time.

## Validation scope

Inline probes record actual branch decisions; the outer wrappers still collect
timing and resource snapshots. Removing the diagnostic probes restores the
original scheduling ASTs, checked by regression tests. Neither layer changes
request selection or scheduling decisions. Original arguments, results
and exceptions are preserved. Counters are owned by one EngineCore scheduler
and reset per log window. The observation path avoids scanning waiting requests
and avoids reading device tensors for diagnostics; optional balance counts are
read only from CPU tensors.

There is still CPU and log-output overhead when enabled. Rate limiting bounds
the log frequency, not the number of instrumented helper calls. Compare enabled
and disabled runs under the same workload before using timing results as a
performance baseline. A final partial window is not force-flushed when scheduling
stops, and summaries can include several request states over time.

CPU tests cover scheduler-result equivalence, real helper failure outcomes,
state reporting, rate limiting, bounded samples, subclass installation,
exception propagation and diagnostic output failures:

```bash
python3 -m unittest discover -s tests/ut/core -p 'test_scheduler_admission*.py' -v
```

NPU validation should run the existing P=4DP×4TP / D=8DP×2TP service with the
debug switch off and on. Compare request correctness, queue/TTFT/TPOT and
throughput; confirm the actual scheduler class and reason fields in both logs.
No NPU performance improvement is claimed by this diagnostic change.
