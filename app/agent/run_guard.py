"""
app/agent/run_guard.py

v3.0 P1 运行态收敛与 HITL 生命周期守护模块：
  - 将「模型/工具活跃推理超时（TURN_TIMEOUT_SEC）」与「人工审批等待时限（APPROVAL_TTL_SEC）」彻底解耦，
    避免用户在 HITL 卡片前停留超过 45 秒被误杀；
  - 提供服务启动与后台看门狗周期巡检（reap_stale_turns_now / RunGuardWatchdog）：
    1. 回收进程异常重启遗留的僵尸 running 轮次，防止前端补发无限轮询；
    2. 回收超过 APPROVAL_TTL_SEC 未审批的 waiting_approval 轮次，原子置 expired 并释放内存 waiter，防泄漏。
"""

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, TypeVar

from app.db import events as event_store

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ApprovalExpiredError(Exception):
    """HITL 待审批等待超过 APPROVAL_TTL_SEC（或已被看门狗提前过期回收）时抛出的内部控制异常。"""


def _read_positive_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "")
    if raw.strip():
        try:
            parsed = float(raw.strip())
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return default


def get_approval_ttl_sec() -> float:
    """HITL 高风险操作待审批最大等待秒数（默认 300 秒 / 5 分钟）。"""
    return _read_positive_float("APPROVAL_TTL_SEC", 300.0)


def get_stale_running_ttl_sec() -> float:
    """运行中僵尸 turn 判定阈值秒数（默认 120 秒）。"""
    return _read_positive_float("STALE_RUNNING_TTL_SEC", 120.0)


def get_run_guard_interval_sec() -> float:
    """后台看门狗周期巡检间隔秒数（默认 30 秒）。"""
    return _read_positive_float("RUN_GUARD_INTERVAL_SEC", 30.0)


async def await_with_active_budget(
    coro: Awaitable[T],
    active_timeout_sec: float,
    is_paused: Callable[[], bool],
    poll_interval_sec: float = 0.2,
) -> T:
    """
    带「暂停冻结」能力的活跃执行预算控制：
      - 当 is_paused() 为 False（正常思考或工具执行）时累计活跃耗时；
      - 当 is_paused() 为 True（HITL 挂起等待人工点击确认）时冻结计时，不消耗 TURN_TIMEOUT_SEC；
      - 一旦活跃累计耗时超出 active_timeout_sec，取消子任务并抛出 asyncio.TimeoutError。
    """
    task: asyncio.Task[T] = asyncio.ensure_future(coro)
    active_elapsed = 0.0
    last_tick = time.monotonic()

    try:
        while not task.done():
            remaining = active_timeout_sec - active_elapsed
            if remaining <= 0 and not is_paused():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.TimeoutError()

            wait_slice = poll_interval_sec if is_paused() else min(poll_interval_sec, max(remaining, 0.01))
            done, _ = await asyncio.wait({task}, timeout=wait_slice)
            now = time.monotonic()
            if not is_paused():
                active_elapsed += now - last_tick
            last_tick = now
            if done:
                return task.result()
        return task.result()
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        raise


def reap_stale_turns_now(
    on_startup: bool = False,
    active_turn_ids: set[str] | None = None,
    on_approval_expired: Callable[[str], None] | None = None,
    approval_ttl_sec: float | None = None,
    running_ttl_sec: float | None = None,
) -> dict[str, int]:
    """
    执行一次同步僵尸 turn 与过期 HITL 审批回收：
      - on_startup=True：服务刚启动，进程内无历史活跃协程，立即回收所有遗留 running 轮次；
      - 周期巡检：仅回收不在 active_turn_ids 且超时未收敛的 running 轮次，以及超出 approval_ttl_sec 的待审批记录；
      - 若某过期审批在内存中仍有挂起 waiter，通过 on_approval_expired 回调唤醒清理，防止协程泄漏。
    """
    eff_approval_ttl = approval_ttl_sec if approval_ttl_sec is not None else get_approval_ttl_sec()
    eff_running_ttl = running_ttl_sec if running_ttl_sec is not None else get_stale_running_ttl_sec()

    result = event_store.reap_stale_turns(
        approval_ttl_sec=eff_approval_ttl,
        running_ttl_sec=eff_running_ttl,
        active_turn_ids=active_turn_ids,
        on_startup=on_startup,
    )
    expired_ids: list[str] = result.get("expired_turn_ids", [])
    if on_approval_expired is not None:
        for tid in expired_ids:
            try:
                on_approval_expired(tid)
            except Exception:
                logger.exception("run_guard 回调通知审批过期失败 turn_id=%s", tid)

    reaped_running = result["reaped_running"]
    expired_approvals = result["expired_approvals"]
    if reaped_running > 0 or expired_approvals > 0:
        logger.warning(
            "run_guard 回收完成 on_startup=%s reaped_running=%d expired_approvals=%d",
            on_startup, reaped_running, expired_approvals,
        )
    return {"reaped_running": reaped_running, "expired_approvals": expired_approvals}


class RunGuardWatchdog:
    """
    轻量后台看门狗：挂载在 FastAPI lifespan 中，周期调用 reap_stale_turns_now。
    不依赖 Celery/APScheduler 等重框架，服务关闭时自动安全取消。
    """

    def __init__(
        self,
        get_active_turn_ids: Callable[[], set[str]],
        on_approval_expired: Callable[[str], None],
        interval_sec: float | None = None,
    ) -> None:
        self._get_active_turn_ids = get_active_turn_ids
        self._on_approval_expired = on_approval_expired
        self._interval_sec = interval_sec if interval_sec is not None else get_run_guard_interval_sec()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """启动时先做一次 on_startup 回收，再开启周期巡检协程。"""
        await asyncio.to_thread(
            reap_stale_turns_now,
            True,
            set(),
            self._on_approval_expired,
        )
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop(), name="run_guard_watchdog")

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval_sec)
                active_ids = self._get_active_turn_ids()
                await asyncio.to_thread(
                    reap_stale_turns_now,
                    False,
                    active_ids,
                    self._on_approval_expired,
                )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("run_guard 后台巡检异常，将在下个周期自动重试")

    async def stop(self) -> None:
        """服务关闭时优雅停止后台看门狗任务。"""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
