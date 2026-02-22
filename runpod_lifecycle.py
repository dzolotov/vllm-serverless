"""
RunPod pod lifecycle manager.

Поднимает pod перед батчем вызовов к vLLM, гасит после.
Используется как контекстный менеджер или напрямую.

Пример:
    from runpod_lifecycle import RunPodLifecycle

    async with RunPodLifecycle() as endpoint:
        # endpoint = "http://1.2.3.4:12345"
        await call_vllm(endpoint, ...)
"""

import asyncio
import logging
import os
import time

import aiohttp

log = logging.getLogger("runpod")

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_POD_ID  = os.environ.get("RUNPOD_POD_ID", "vjsfro8wj8hcx2")
GRAPHQL_URL    = "https://api.runpod.io/graphql"

HEALTH_TIMEOUT  = 300   # сек ожидания готовности пода
HEALTH_INTERVAL = 5     # сек между проверками


class RunPodLifecycle:
    """Async context manager: start pod → yield endpoint → stop pod."""

    def __init__(
        self,
        api_key: str = RUNPOD_API_KEY,
        pod_id: str  = RUNPOD_POD_ID,
    ) -> None:
        self.api_key  = api_key
        self.pod_id   = pod_id
        self.endpoint = ""
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> str:
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        await self._start()
        self.endpoint = await self._wait_for_endpoint()
        await self._wait_for_vllm()
        return self.endpoint

    async def __aexit__(self, *_) -> None:
        try:
            await self._stop()
        finally:
            if self._session:
                await self._session.close()

    # ── RunPod GraphQL ────────────────────────────────────────────────────────

    async def _graphql(self, query: str, variables: dict) -> dict:
        assert self._session
        async with self._session.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as r:
            return await r.json()

    async def _start(self) -> None:
        mutation = """
        mutation startPod($id: String!) {
          podStart(input: { podId: $id }) { id desiredStatus }
        }
        """
        data = await self._graphql(mutation, {"id": self.pod_id})
        status = data["data"]["podStart"]["desiredStatus"]
        log.info(f"[runpod] pod {self.pod_id} start requested (status={status})")

    async def _stop(self) -> None:
        mutation = """
        mutation stopPod($id: String!) {
          podStop(input: { podId: $id }) { id desiredStatus }
        }
        """
        data = await self._graphql(mutation, {"id": self.pod_id})
        status = data["data"]["podStop"]["desiredStatus"]
        log.info(f"[runpod] pod {self.pod_id} stopped (status={status})")

    async def _wait_for_endpoint(self) -> str:
        query = """
        query getPod($id: String!) {
          pod(input: { podId: $id }) {
            runtime { ports { ip privatePort publicPort } }
          }
        }
        """
        deadline = time.monotonic() + HEALTH_TIMEOUT
        while time.monotonic() < deadline:
            data = await self._graphql(query, {"id": self.pod_id})
            ports = (data["data"]["pod"].get("runtime") or {}).get("ports") or []
            port = next((p for p in ports if p["privatePort"] == 8000), None)
            if port:
                endpoint = f"http://{port['ip']}:{port['publicPort']}"
                log.info(f"[runpod] endpoint ready: {endpoint}")
                return endpoint
            log.debug("[runpod] waiting for port 8000...")
            await asyncio.sleep(HEALTH_INTERVAL)
        raise TimeoutError(f"Pod did not expose port 8000 in {HEALTH_TIMEOUT}s")

    async def _wait_for_vllm(self) -> None:
        url = f"{self.endpoint}/health"
        deadline = time.monotonic() + HEALTH_TIMEOUT
        assert self._session
        while time.monotonic() < deadline:
            try:
                async with self._session.get(
                    url, timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status == 200:
                        log.info("[runpod] vLLM ready")
                        return
            except Exception:
                pass
            log.debug("[runpod] waiting for vLLM /health...")
            await asyncio.sleep(HEALTH_INTERVAL)
        raise TimeoutError(f"vLLM did not become ready in {HEALTH_TIMEOUT}s")


# ── sync helper для использования из синхронного кода (llm_eval.py) ──────────

_active_endpoint: str = ""
_active_loop: asyncio.AbstractEventLoop | None = None
_active_session: aiohttp.ClientSession | None = None


def _get_lifecycle() -> RunPodLifecycle:
    return RunPodLifecycle(api_key=RUNPOD_API_KEY, pod_id=RUNPOD_POD_ID)


def ensure_running() -> str:
    """
    Синхронно поднимает pod если ещё не запущен.
    Возвращает endpoint URL. Вызывай перед батчем.
    """
    global _active_endpoint, _active_loop, _active_session
    if _active_endpoint:
        return _active_endpoint

    if not RUNPOD_API_KEY:
        log.warning("[runpod] RUNPOD_API_KEY not set, using ORACLE_VLLM_URL as-is")
        return os.environ.get("ORACLE_VLLM_URL", "http://localhost:8000")

    lc = _get_lifecycle()
    loop = asyncio.new_event_loop()

    async def _run() -> str:
        session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {lc.api_key}"}
        )
        lc._session = session
        await lc._start()
        endpoint = await lc._wait_for_endpoint()
        await lc._wait_for_vllm()
        return endpoint

    _active_endpoint = loop.run_until_complete(_run())
    _active_loop = loop
    return _active_endpoint


def shutdown() -> None:
    """Синхронно гасит pod. Вызывай после батча."""
    global _active_endpoint, _active_loop, _active_session
    if not _active_loop or not _active_endpoint:
        return

    lc = _get_lifecycle()
    loop = _active_loop

    async def _run() -> None:
        session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {lc.api_key}"}
        )
        lc._session = session
        try:
            await lc._stop()
        finally:
            await session.close()

    try:
        loop.run_until_complete(_run())
    except Exception as e:
        log.warning(f"[runpod] shutdown error: {e}")
    finally:
        _active_endpoint = ""
        _active_loop = None
