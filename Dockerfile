# syntax=docker/dockerfile:1.4
ARG VLLM_VERSION=0.9.1
FROM vllm/vllm-openai:v${VLLM_VERSION} AS base

# ── модель ──────────────────────────────────────────────────────────────────
ARG MODEL_ID=Qwen/Qwen3-32B-FP8
ARG MODEL_REVISION=main
ARG HF_TOKEN=""
ENV MODEL_ID=${MODEL_ID}
ENV MODEL_PATH=/model

# ── скачиваем веса при сборке ────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/huggingface \
    huggingface-cli download \
        --repo-type model \
        --revision ${MODEL_REVISION} \
        ${HF_TOKEN:+--token ${HF_TOKEN}} \
        --local-dir ${MODEL_PATH} \
        ${MODEL_ID}

# ── параметры сервера ────────────────────────────────────────────────────────
ENV VLLM_HOST=0.0.0.0
ENV VLLM_PORT=8000
ENV MAX_MODEL_LEN=32768
ENV GPU_MEMORY_UTILIZATION=0.95
ENV MAX_NUM_SEQS=32
ENV TENSOR_PARALLEL_SIZE=1

# warmup намеренно не делается — образ стартует быстро,
# прогрев CUDA-графов происходит на первом реальном запросе на GPU-машине

# ── entrypoint ───────────────────────────────────────────────────────────────
EXPOSE ${VLLM_PORT}

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "/model", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--max-model-len", "32768", \
     "--gpu-memory-utilization", "0.95", \
     "--max-num-seqs", "32", \
     "--tensor-parallel-size", "1", \
     "--served-model-name", "qwen3", \
     "--enable-reasoning", \
     "--reasoning-parser", "deepseek_r1", \
     "--enable-prefix-caching", \
     "--disable-log-requests"]
