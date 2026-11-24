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

The interval is a positive integer in milliseconds, measured independently per
EngineCore. The default is 1000 ms. Each summary includes at most four event
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
| `observations` | Counts of actual helper outcomes. These are call/event counts, not unique denied requests. |
| `step_limits` | Counts of steps with backlog and an observed capacity or policy limit at return. These are not attributed to individual requests. |
| `helper_timings` | Call count, total and maximum duration of each observed helper. |
| `samples` | Up to four observed events with internal RequestID, status and waiting/running phase. |
| `omitted_samples` | Events omitted from the bounded sample list; still included in counters. |

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
not proof of which branch prevented every waiting request from running. Other
reasons, such as LoRA limits, encoder limits, long-prefill policies and request
ordering, are not exhaustively identified by this diagnostic. An empty reason
counter therefore does not mean there was no scheduling constraint.

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

The wrappers call the existing scheduler implementation and owned helpers;
they do not copy or replace scheduling decisions. Original arguments, results
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
python3 tests/ut/core/test_scheduler_admission_diagnostics.py
```

NPU validation should run the existing P=4DP×4TP / D=8DP×2TP service with the
debug switch off and on. Compare request correctness, queue/TTFT/TPOT and
throughput; confirm the actual scheduler class and reason fields in both logs.
No NPU performance improvement is claimed by this diagnostic change.
