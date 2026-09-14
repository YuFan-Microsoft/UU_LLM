# ============================================================================
# SIDReasoner runtime image - built on the official verl vLLM 0.24 image.
#
# Base tag: verlai/verl:vllm024.dev2
#   (Ubuntu 24.04 + CUDA 13.0.2 + Python 3.12)
#
# The base image ALREADY ships the following — DO NOT reinstall them
# (reinstalling torch in particular would break the whole CUDA/vLLM stack):
#   - CUDA 13.0.2 and cuDNN 9
#   - torch==2.11.0 (cu130), vllm==0.24.0
#   - transformers==5.3.0, flash-attn==2.8.3
#   - Apex, TransformerEngine, Megatron-Bridge and DeepEP
#
# On top of the base we add ONLY what is missing for this project:
#   - deepspeed     (Stage-1/2 SFT: phase1/phase2 use an explicit DeepSpeed loop)
#   - fire          (evaluation + SFT launcher CLIs)
#   - tqdm and the pinned Azure SDK packages used by project integrations
#   - GitHub Copilot CLI
#
# NOTE: verl itself is NOT pip-installed. We run the vendored ./verl_060 from
#       source. That source must be migrated if its vLLM 0.10 compatibility
#       branches do not support vLLM 0.24.
# HOST: CUDA 13 requires an NVIDIA R580+ driver. A100 supports this stack.
# ============================================================================
FROM verlai/verl:vllm024.dev2

ENV PIP_INDEX_URL=https://pypi.org/simple

# (optional) git-lfs for pulling HuggingFace checkpoints + shell tools
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl git-lfs tmux unzip vim && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://gh.io/copilot-install | bash && \
    copilot --version

# --- Project deps NOT covered by the base image or by vLLM ---
#   deepspeed : Stage-1/2 SFT (phase1_alignment_sft / phase2_reasoning_activation)
#   fire      : evaluation scripts + SFT launchers
RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    deepspeed \
    fire \
    ipykernel \
    tqdm \
    azure-core==1.30.1 \
    azure-identity==1.16.0 \
    azure-keyvault==4.2.0 \
    azure-keyvault-certificates==4.8.0 \
    azure-keyvault-keys==4.9.0 \
    azure-keyvault-secrets==4.8.0
RUN pip uninstall -y nvtx || true

# --- Sanity check: project installs must not disturb the base stack ---
RUN python -c "from importlib.metadata import version as v; \
print('torch', v('torch'), '| vllm', v('vllm'), '| deepspeed', v('deepspeed'), '| ipykernel', v('ipykernel'), '| transformers', v('transformers')); \
assert v('torch').startswith('2.11'), 'torch got overridden: ' + v('torch'); \
assert v('vllm') == '0.24.0', 'unexpected vllm: ' + v('vllm'); \
assert v('transformers') == '5.5.3', 'unexpected transformers: ' + v('transformers')"