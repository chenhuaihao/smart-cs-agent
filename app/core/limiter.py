"""
app/core/limiter.py

轻量级内存滑动窗口限流模块（纯 Python 标准库实现）：
- 零外部依赖：仅使用 time, threading, collections, os；
- 兼容反向代理：优先从 X-Forwarded-For 请求头解析客户端真实 IP；
- 滑动窗口算法：以 60 秒为滑动时间窗口，精确记录并统计每 IP 的请求时间戳；
- 环境变量配置：支持通过 RATE_LIMIT_PER_MINUTE 动态配置（默认 15 次/分钟）；
- 超限友好提示：超限时抛出 HTTP 429 Too Many Requests，附带 Retry-After 标头与中文友好 JSON 提示；
- 内存防爆机制：内置惰性过期淘汰机制，防止高并发场景下内存无限膨胀。
"""

import os
import threading
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request, status


class SlidingWindowLimiter:
    """
    基于内存的滑动窗口限流器。
    
    使用 collections.defaultdict(list) 存储每个客户端 IP 在滑动窗口内的请求时间戳。
    使用 threading.Lock 保证并发读写安全性。
    """

    def __init__(self, window_seconds: int = 60, default_limit: int = 15) -> None:
        self.window_seconds = window_seconds
        self.default_limit = default_limit
        # 记录各 IP 的历史请求时间戳列表: {ip: [ts1, ts2, ...]}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_limit(self) -> int:
        """
        动态读取环境变量 RATE_LIMIT_PER_MINUTE，支持热调整或环境覆盖。
        若未配置或解析失败，回退至 default_limit (15)。
        """
        env_val = os.getenv("RATE_LIMIT_PER_MINUTE")
        if env_val:
            try:
                parsed = int(env_val.strip())
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return self.default_limit

    def check(self, client_ip: str) -> None:
        """
        检查指定客户端 IP 是否超限：
        1. 剔除超过滑动窗口 (当前时间 - window_seconds) 的历史请求；
        2. 若窗口内请求数已达到上限，计算需等待的秒数，抛出 HTTP 429；
        3. 若未超限，记录本次请求时间戳；
        4. 定期惰性清理长期未活跃 IP，防止内存泄漏。
        """
        now = time.time()
        limit = self._get_limit()
        window_start = now - self.window_seconds

        with self._lock:
            # 1. 过滤掉滑动窗口之外的过期时间戳
            timestamps = self._history[client_ip]
            valid_timestamps = [t for t in timestamps if t > window_start]

            # 2. 判定是否超出每分钟限额
            if len(valid_timestamps) >= limit:
                # 计算窗口内最早一笔请求移出窗口还需等待的秒数
                earliest_ts = valid_timestamps[0]
                retry_after = max(1, int(self.window_seconds - (now - earliest_ts)) + 1)
                self._history[client_ip] = valid_timestamps
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=(
                        f"请求过于频繁，请稍后再试。系统当前限制每分钟最多请求 {limit} 次"
                        f"（客户端 IP: {client_ip}，请在 {retry_after} 秒后重试）。"
                    ),
                    headers={"Retry-After": str(retry_after)},
                )

            # 3. 未超限，追加当前时间戳并更新
            valid_timestamps.append(now)
            self._history[client_ip] = valid_timestamps

            # 4. 内存防护：当记录的 IP 数量超过 5000 时，清理全部已过期的 IP 键
            if len(self._history) > 5000:
                self._cleanup_expired(now)

    def _cleanup_expired(self, now: float) -> None:
        """
        遍历并清理长期没有新请求的过期 IP 记录，释放内存字典空间。
        """
        window_start = now - self.window_seconds
        stale_ips = [
            ip for ip, ts_list in self._history.items()
            if not ts_list or ts_list[-1] <= window_start
        ]
        for ip in stale_ips:
            del self._history[ip]


# 模块级全局单例限流器
_chat_limiter = SlidingWindowLimiter()


def get_client_ip(request: Request) -> str:
    """
    解析请求的真实客户端 IP：
    1. 优先检查反向代理（如 Nginx、Traefik、SLB）透传的 X-Forwarded-For 请求头；
    2. 若存在多级代理，取最左侧的原始客户端 IP；
    3. 若无该请求头，取底层 socket 的 request.client.host；
    4. 若仍为空则回退至 '127.0.0.1'。
    """
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # X-Forwarded-For: <client>, <proxy1>, <proxy2>
        client_ip = x_forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip

    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


def check_chat_rate_limit(request: Request) -> None:
    """
    挂载在 POST /api/chat 的 FastAPI 依赖项函数。
    在执行任何 Agent 思考或大模型 API 调用前进行防刷拦截。
    """
    client_ip = get_client_ip(request)
    _chat_limiter.check(client_ip)
