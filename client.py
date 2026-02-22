"""
Serverless batch client для Qwen3 на RunPod.

Жизненный цикл:
  1. Поднять pod через RunPod API
  2. Дождаться готовности vLLM (/health)
  3. Отправить батч по 1000 записей (параллельные запросы)
  4. Погасить pod

Использование:
  export RUNPOD_API_KEY=...
  export RUNPOD_POD_ID=...           # ID существующего (остановленного) пода
  python client.py input.jsonl output.jsonl
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNPOD_POD_ID  = os.environ["RUNPOD_POD_ID"]

BATCH_SIZE        = 1000
CONCURRENCY       = 32          # параллельных запросов к vLLM
HEALTH_TIMEOUT    = 300         # сек ожидания старта пода
HEALTH_INTERVAL   = 5           # сек между проверками /health
IDLE_SHUTDOWN_SEC = 30          # сколько ждать после последнего запроса перед гашением

RUNPOD_API = "https://api.runpod.io/v2"
GRAPHQL_URL = "https://api.runpod.io/graphql"

# ── RunPod API ────────────────────────────────────────────────────────────────

async def pod_start(session: aiohttp.ClientSession) -> str:
    """Запускает pod и возвращает его публичный IP:port."""
    mutation = """
    mutation startPod($id: String!) {
      podStart(input: { podId: $id }) {
        id
        desiredStatus
        runtime { ports { ip privatePort publicPort type } }
      }
    }
    """
    async with session.post(
        GRAPHQL_URL,
        json={"query": mutation, "variables": {"id": RUNPOD_POD_ID}},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    ) as r:
        data = await r.json()
    pod = data["data"]["podStart"]
    print(f"[runpod] pod {pod['id']} starting (status={pod['desiredStatus']})")
    return pod["id"]


async def pod_get_endpoint(session: aiohttp.ClientSession) -> str:
    """Опрашивает RunPod пока pod не запустится, возвращает http://ip:port."""
    query = """
    query getPod($id: String!) {
      pod(input: { podId: $id }) {
        id
        desiredStatus
        runtime { ports { ip privatePort publicPort type } }
      }
    }
    """
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        async with session.post(
            GRAPHQL_URL,
            json={"query": query, "variables": {"id": RUNPOD_POD_ID}},
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        ) as r:
            data = await r.json()
        pod = data["data"]["pod"]
        runtime = pod.get("runtime") or {}
        ports = runtime.get("ports") or []
        http_port = next((p for p in ports if p["privatePort"] == 8000), None)
        if http_port:
            endpoint = f"http://{http_port['ip']}:{http_port['publicPort']}"
            print(f"[runpod] pod endpoint: {endpoint}")
            return endpoint
        print(f"[runpod] waiting for pod to expose port 8000...")
        await asyncio.sleep(HEALTH_INTERVAL)
    raise TimeoutError(f"Pod did not expose port 8000 in {HEALTH_TIMEOUT}s")


async def pod_stop(session: aiohttp.ClientSession) -> None:
    mutation = """
    mutation stopPod($id: String!) {
      podStop(input: { podId: $id }) { id desiredStatus }
    }
    """
    async with session.post(
        GRAPHQL_URL,
        json={"query": mutation, "variables": {"id": RUNPOD_POD_ID}},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    ) as r:
        data = await r.json()
    status = data["data"]["podStop"]["desiredStatus"]
    print(f"[runpod] pod stopped (desiredStatus={status})")


# ── vLLM health ───────────────────────────────────────────────────────────────

async def wait_for_vllm(session: aiohttp.ClientSession, endpoint: str) -> None:
    url = f"{endpoint}/health"
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 200:
                    print("[vllm] server is ready")
                    return
        except Exception:
            pass
        print("[vllm] waiting for server...")
        await asyncio.sleep(HEALTH_INTERVAL)
    raise TimeoutError(f"vLLM did not become ready in {HEALTH_TIMEOUT}s")


# ── inference ─────────────────────────────────────────────────────────────────

async def infer_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    sem: asyncio.Semaphore,
    idx: int,
    record: dict,
) -> dict:
    """Отправляет один запрос, возвращает запись с добавленным полем 'output'."""
    messages = record.get("messages") or [{"role": "user", "content": record["text"]}]
    payload = {
        "model": "qwen3",
        "messages": messages,
        "temperature": record.get("temperature", 0.6),
        "max_tokens": record.get("max_tokens", 2048),
    }
    async with sem:
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as r:
            data = await r.json()
    result = record.copy()
    choice = data["choices"][0]["message"]
    result["output"] = choice.get("content", "")
    result["reasoning"] = choice.get("reasoning_content", "")
    result["_idx"] = idx
    return result


async def process_batch(
    session: aiohttp.ClientSession,
    endpoint: str,
    records: list[dict],
) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        infer_one(session, endpoint, sem, i, rec)
        for i, rec in enumerate(records)
    ]
    results = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        done += 1
        if done % 100 == 0 or done == len(records):
            print(f"[batch] {done}/{len(records)} done")
    results.sort(key=lambda r: r["_idx"])
    for r in results:
        del r["_idx"]
    return results


# ── main ──────────────────────────────────────────────────────────────────────

async def main(input_path: str, output_path: str) -> None:
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[main] loaded {len(records)} records from {input_path}")

    async with aiohttp.ClientSession() as session:
        # поднимаем pod
        await pod_start(session)
        endpoint = await pod_get_endpoint(session)
        await wait_for_vllm(session, endpoint)

        # обрабатываем батчами по 1000
        all_results = []
        for batch_start in range(0, len(records), BATCH_SIZE):
            batch = records[batch_start : batch_start + BATCH_SIZE]
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[main] batch {batch_num}/{total_batches} ({len(batch)} records)")
            results = await process_batch(session, endpoint, batch)
            all_results.extend(results)

        # пишем результат
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[main] written {len(all_results)} records to {output_path}")

        # гасим pod
        await pod_stop(session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input",  help="Input JSONL file (fields: text OR messages[])")
    parser.add_argument("output", help="Output JSONL file")
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output))
