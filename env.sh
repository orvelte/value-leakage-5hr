# Source before any GPU work: `source env.sh`
#
# WHY: this pod's kernel driver is 550.127.05 (CUDA 12.4), but vllm==0.27.1 pins torch 2.13.0,
# whose PyPI wheels are CUDA 13 builds. CUDA forward compatibility (cuda-compat-13-0, driver
# userspace 580.178.04) bridges the gap — supported on data-center GPUs, which the A100 is.
# Without this the cu13 wheels fail at init with "CUDA driver version is insufficient".
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}

# Model + lens weights live on the /workspace network volume (197T free, ~630 MB/s), not the
# 100G container overlay, which the CUDA-13 wheel stack alone fills a third of.
export HF_HOME=/workspace/.cache/huggingface/
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
