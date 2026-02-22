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

# ── прогрев CUDA-графов при сборке ──────────────────────────────────────────
# Запускаем vLLM в режиме одного прогревочного шага — компилирует CUDA-графы
# и сохраняет кэш torch.compile в /opt/vllm_cache.
# При старте контейнера кэш подхватывается без перекомпиляции.
ENV VLLM_CACHE_ROOT=/opt/vllm_cache
RUN --mount=type=cache,target=/root/.cache/huggingface \
    mkdir -p ${VLLM_CACHE_ROOT} && \
    python -c "
import os, torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

llm = LLM(
    model=os.environ['MODEL_PATH'],
    max_model_len=int(os.environ['MAX_MODEL_LEN']),
    gpu_memory_utilization=float(os.environ['GPU_MEMORY_UTILIZATION']),
    tensor_parallel_size=int(os.environ['TENSOR_PARALLEL_SIZE']),
    enforce_eager=False,          # строим CUDA-графы
    max_num_seqs=int(os.environ['MAX_NUM_SEQS']),
    download_dir=os.environ['MODEL_PATH'],
    load_format='auto',
    enable_prefix_caching=True,   # соответствует серверному флагу
)

# прогрев: thinking=True соответствует --enable-reasoning / --reasoning-parser deepseek_r1
# токенизируем с chat_template чтобы прогреть тот же путь, что и на сервере
tokenizer = get_tokenizer(os.environ['MODEL_PATH'])
messages = [{'role': 'user', 'content': 'Hello'}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,         # активирует <think>...</think> путь
)
params = SamplingParams(temperature=0, max_tokens=1)
llm.generate([prompt], params)
print('Warmup complete')
del llm
torch.cuda.empty_cache()
" || echo "GPU not available during build — skipping warmup"

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
