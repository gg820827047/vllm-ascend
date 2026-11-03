#!/usr/bin/env bash
set -euo pipefail

# Derived from the pinned official 0.32 CI config by this report.
# Statically checked; not executed on A3 in this workspace.
: "${MODEL_PATH:?Set MODEL_PATH to the W8A8 checkpoint directory}"
: "${LOCAL_IP:?Set LOCAL_IP to this node service IP}"
: "${NIC_NAME:?Set NIC_NAME to this node network interface}"
[[ $# -eq 8 ]] || { echo "Expected: devices port dp rank address rpc tp pcp" >&2; exit 2; }
export ASCEND_RT_VISIBLE_DEVICES="$1"
export SERVER_PORT="$2"
export DP_SIZE="$3"
export DP_RANK="$4"
export DP_ADDRESS="$5"
export DP_RPC_PORT="$6"
export TP_SIZE="$7"
export CP_SIZE="$8"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export VLLM_USE_MODELSCOPE=true
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_ENGINE_READY_TIMEOUT_S=360000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export https_proxy=''
export http_proxy=''
export all_proxy=''
export HTTPS_PROXY=''
export HTTP_PROXY=''
export ALL_PROXY=''
export HCCL_BUFFSIZE=1024
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export no_proxy="127.0.0.1,localhost,${LOCAL_IP},${PREFILL_IP:-},${DECODE_IP:-}"
export NO_PROXY="$no_proxy"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"

exec vllm serve "$MODEL_PATH" \
    --served-model-name dsv41 \
    --host 0.0.0.0 \
    --port "$SERVER_PORT" \
    --data-parallel-size "${DP_SIZE}" \
    --data-parallel-rank "${DP_RANK}" \
    --data-parallel-address "${DP_ADDRESS}" \
    --data-parallel-rpc-port "${DP_RPC_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --prefill-context-parallel-size "${CP_SIZE}" \
    --enable-expert-parallel \
    --seed 1024 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
    --trust-remote-code \
    --block-size 128 \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":32}' \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.92 \
    --quantization ascend \
    --enforce-eager \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
    --additional-config '{"enable_cpu_binding":true,"enable_fused_mc2":1,"enable_flashcomm1":true,"enable_shared_expert_dp":true}' \
    --kv-transfer-config '{"kv_connector":"MooncakeConnectorV2","kv_role":"kv_producer","kv_port":"30000","engine_id":"0","kv_connector_extra_config":{"prefill":{"dp_size":2,"tp_size":2},"decode":{"dp_size":8,"tp_size":2}}}'
