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
#   - flash-linear-attention + causal-conv1d (Qwen3.5 linear-attention fast path,
#     about 10x SFT throughput; see code/trainer/SFT/README.md)
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

# --- Qwen3.5 linear-attention fast path (24 Gated DeltaNet layers) ---
# Without these, transformers silently falls back to PyTorch Gated Delta Rule / Conv1D (~10x slower SFT);
# the SFT trainer refuses to start unless both are importable.
#   flash-linear-attention / fla-core: pure-Python Triton kernels. --no-deps keeps torch/transformers/triton
#     from the base image (fla-core only needs einops besides those).
#   causal-conv1d: no prebuilt wheel exists for torch 2.11, so build it. Its setup.py hardcodes ~9 GPU
#     architectures (SM75..SM121) and ignores TORCH_CUDA_ARCH_LIST, so patch it to the target arch only.
#     Default 80 = A100; set --build-arg CAUSAL_CONV1D_CUDA_ARCH=90 for H100, etc.
ARG CAUSAL_CONV1D_CUDA_ARCH=80
ARG CAUSAL_CONV1D_MAX_JOBS=16
RUN pip install --no-cache-dir einops ninja && \
    pip install --no-cache-dir --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
RUN set -eux; \
    build_dir="$(mktemp -d)"; cd "$build_dir"; \
    curl -fsSL https://files.pythonhosted.org/packages/source/c/causal-conv1d/causal_conv1d-1.7.0.tar.gz | tar xz; \
    cd causal_conv1d-1.7.0; \
    python -c 'import os; path = "setup.py"; source = open(path).read(); \
anchor = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"; \
assert anchor in source and "arch=compute_80,code=sm_80" in source, "causal-conv1d setup.py layout changed"; \
arch = os.environ["CAUSAL_CONV1D_CUDA_ARCH"]; \
override = "    cc_flag = [f for f in cc_flag if f != \"-gencode\" and not f.startswith(\"arch=compute_\")]\n" \
"    cc_flag += [\"-gencode\", \"arch=compute_%s,code=sm_%s\"]\n" % (arch, arch); \
open(path, "w").write(source.replace(anchor, override + anchor, 1))'; \
    grep -n -A1 'cc_flag = \[f for f' setup.py; \
    CAUSAL_CONV1D_FORCE_BUILD=TRUE MAX_JOBS="${CAUSAL_CONV1D_MAX_JOBS}" \
        pip install --no-cache-dir --no-build-isolation --no-deps . ; \
    cd /; rm -rf "$build_dir"

# --- Sanity check: project installs must not disturb the base stack ---
RUN python -c "from importlib.metadata import version as v; \
print('torch', v('torch'), '| vllm', v('vllm'), '| deepspeed', v('deepspeed'), '| ipykernel', v('ipykernel'), '| transformers', v('transformers')); \
assert v('torch').startswith('2.11'), 'torch got overridden: ' + v('torch'); \
assert v('vllm') == '0.24.0', 'unexpected vllm: ' + v('vllm'); \
assert v('transformers') == '5.5.3', 'unexpected transformers: ' + v('transformers')"
# Import checks need a GPU, so only verify the packages and the compiled CUDA extension are present here; the
# SFT trainer checks is_flash_linear_attention_available() / is_causal_conv1d_available() on every rank.
RUN python -c "from importlib.metadata import version as v; import importlib.util as u; \
print('fla-core', v('fla-core'), '| flash-linear-attention', v('flash-linear-attention'), '| causal-conv1d', v('causal-conv1d')); \
assert v('flash-linear-attention') == '0.5.2' and v('fla-core') == '0.5.2', 'unexpected FLA version'; \
assert v('causal-conv1d') == '1.7.0', 'unexpected causal-conv1d: ' + v('causal-conv1d'); \
assert u.find_spec('causal_conv1d_cuda') is not None, 'causal_conv1d_cuda extension was not built'"