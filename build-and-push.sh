#!/usr/bin/env bash
set -euo pipefail

IMAGE="dmitriizolotov/qwen3-serverless"
TAG="qwen3-32b-fp8"
FULL_IMAGE="${IMAGE}:${TAG}"

# ── проверка окружения ───────────────────────────────────────────────────────
echo "==> Checking environment..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
docker --version
echo ""

# ── логин в Docker Hub ───────────────────────────────────────────────────────
if [[ -z "${DOCKER_PASSWORD:-}" ]]; then
    echo "==> Docker Hub login (enter password when prompted):"
    docker login -u dmitriizolotov
else
    echo "==> Docker Hub login (from env)..."
    echo "${DOCKER_PASSWORD}" | docker login -u dmitriizolotov --password-stdin
fi

# ── копируем проект если запущено не из его директории ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── сборка ───────────────────────────────────────────────────────────────────
echo ""
echo "==> Building ${FULL_IMAGE}..."
echo "    This will download model weights (~20GB) and compile CUDA graphs."
echo "    Expected time: 20-40 min depending on network and GPU."
echo ""

docker build \
    --progress=plain \
    --build-arg MODEL_ID=Qwen/Qwen3-32B-FP8 \
    ${HF_TOKEN:+--build-arg HF_TOKEN="${HF_TOKEN}"} \
    -t "${FULL_IMAGE}" \
    -t "${IMAGE}:latest" \
    .

# ── пуш ─────────────────────────────────────────────────────────────────────
echo ""
echo "==> Pushing ${FULL_IMAGE}..."
docker push "${FULL_IMAGE}"
docker push "${IMAGE}:latest"

echo ""
echo "==> Done! Image available at:"
echo "    docker pull ${FULL_IMAGE}"
