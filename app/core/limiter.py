"""
app/core/limiter.py

轻量级内存限流与用量熔断模块（纯 Python 标准库实现）：
- 零外部依赖：仅使用 time, threading, collections, os, datetime；
- 兼容反向代理：优先从 X-Forwarded-For（首个有效 IP）及 X-Real-IP 解析客户端真实 IP；
- 单 IP 滑动窗口限流：以 60 秒为滑动时间窗口，精确记录并统计每 IP 的请求时间戳；
- 全局每日用量熔断：按北京时间（UTC+8）每日 00:00 自动重置计数，防止公网演示被刷爆；
- 环境变量配置：
    - RATE_LIMIT_PER_MINUTE（默认 15 次/分钟/IP）
    - DAILY_CHAT_LIMIT（默认 200 次/日全局）
- 超限友好提示：超限时抛出 HTTP 429 Too Many Requests，错误信息中脱敏去除了客户端真实 IP；
- 内存防爆机制：内置惰性过期淘汰机制，防止高并发场景下内存无限膨胀。
"""

import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from fastapi import HTTPException, Request, status

# 北京时间时区定义 (UTC+8)
BEIJING_TZ = timezone(timedelta(hours=8))


# ==============================================================================
# 1. 单 IP 滑动窗口限流器 (SlidingWindowLimiter)
# ==============================================================================

class SlidingWindowLimiter:
    """
    基于内存的滑动窗口限流器。
    
    使用 collections.defaultdict(list) 存储每个客户端 IP 在滑动窗口内的请求时间戳。
    使用 threading.Lock 保证并发读写安全性。
    """

    def __init__(
        self,
        window_seconds: int = 60,
        default_limit: int = 15,
        env_var: str = "RATE_LIMIT_PER_MINUTE",
    ) -> None:
        self.window_seconds = window_seconds
        self.default_limit = default_limit
        self.env_var = env_var
        # 记录各 IP 的历史请求时间戳列表: {ip: [ts1, ts2, ...]}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_limit(self) -> int:
        """
        动态读取对应环境变量（如 RATE_LIMIT_PER_MINUTE / API_READ_RATE_LIMIT_PER_MINUTE），
        若未配置或解析失败，回退至 default_limit。
        """
        env_val = os.getenv(self.env_var)
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
        2. 若窗口内请求数已达到上限，计算需等待的秒数，抛出 HTTP 429（不暴露客户端 IP）；
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
                    detail=f"请求过于频繁，请稍后再试（每分钟限制 {limit} 次，请在 {retry_after} 秒后重试）。",
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


# 模块级单 IP 限流单例（对话 / 业务只读 / 业务写操作分桶隔离）
_chat_limiter = SlidingWindowLimiter(window_seconds=60, default_limit=15, env_var="RATE_LIMIT_PER_MINUTE")
_api_read_limiter = SlidingWindowLimiter(
    window_seconds=60,
    default_limit=60,
    env_var="API_READ_RATE_LIMIT_PER_MINUTE",
)
_api_write_limiter = SlidingWindowLimiter(
    window_seconds=60,
    default_limit=20,
    env_var="API_WRITE_RATE_LIMIT_PER_MINUTE",
)


# ==============================================================================
# 2. 每日全局 + 单 IP 调用用量熔断器 (DailyQuotaLimiter)
# ==============================================================================

class DailyQuotaLimiter:
    """
    基于内存的每日双层用量熔断限流器（v3.0 收尾加固）：

    - 按北京时间（UTC+8）自然日 00:00 自动重置当日计数器；
    - 全局日上限：读取 DAILY_CHAT_LIMIT，默认 200 次/天；
    - 单 IP 日上限：读取 DAILY_IP_CHAT_LIMIT，默认 30 次/天/IP，防止单个刷量者耗尽全站演示额度
      导致后续真实面试官无法体验（若显式调高 DAILY_CHAT_LIMIT 用于压测且未单独设单 IP 上限，则放宽至全局上限）；
    - 采用 threading.Lock 保证并发读写安全性。
    """

    def __init__(self, default_limit: int = 200, default_ip_limit: int = 30) -> None:
        self.default_limit = default_limit
        self.default_ip_limit = default_ip_limit
        self._current_date_str: str = ""
        self._count: int = 0
        self._ip_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def _get_limit(self) -> int:
        """动态读取环境变量 DAILY_CHAT_LIMIT，支持覆盖默认全局日上限。"""
        env_val = os.getenv("DAILY_CHAT_LIMIT")
        if env_val:
            try:
                parsed = int(env_val.strip())
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return self.default_limit

    def _get_ip_limit(self, global_limit: int) -> int:
        """动态读取 DAILY_IP_CHAT_LIMIT；若用于评测显式大幅调高了全局上限且未设单 IP 限制，则同步放宽。"""
        env_val = os.getenv("DAILY_IP_CHAT_LIMIT")
        if env_val:
            try:
                parsed = int(env_val.strip())
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        if global_limit > self.default_limit:
            return global_limit
        return min(self.default_ip_limit, global_limit)

    def check_and_increment(self, client_ip: str | None = None) -> None:
        """
        判定是否超出单 IP 日上限或当日全局总配额：
        - 跨日（北京时间 00:00）自动重置当日计数；
        - 超限抛出 HTTP 429 熔断异常；
        - 未超限则同步自增全局与单 IP 计数并放行。
        """
        now_beijing = datetime.now(BEIJING_TZ)
        today_str = now_beijing.strftime("%Y-%m-%d")
        limit = self._get_limit()
        ip_limit = self._get_ip_limit(limit)

        with self._lock:
            # 跨自然日自动归零重置
            if today_str != self._current_date_str:
                self._current_date_str = today_str
                self._count = 0
                self._ip_counts.clear()

            # 1. 先检查单 IP 日上限，防止单访客刷光全站额度
            if client_ip and self._ip_counts[client_ip] >= ip_limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"您今日的演示体验次数已达上限（单 IP 每日限 {ip_limit} 次），请明日 0 点后再试。",
                )

            # 2. 再检查全站每日总限额
            if self._count >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="今日公网演示总额度已用完，请明日 0 点后再试。",
                )

            self._count += 1
            if client_ip:
                self._ip_counts[client_ip] += 1

    def get_stats(self) -> dict:
        """获取当前配额使用快照（用于诊断与监控）"""
        with self._lock:
            limit = self._get_limit()
            return {
                "date": self._current_date_str,
                "used": self._count,
                "limit": limit,
                "ip_limit": self._get_ip_limit(limit),
                "active_ips": len(self._ip_counts),
            }


# 模块级每日全局配额熔断器单例
_daily_quota_limiter = DailyQuotaLimiter()


# ==============================================================================
# 3. 辅助函数与 FastAPI 依赖项
# ==============================================================================

def _is_proxy_trusted() -> bool:
    """
    是否信任反向代理透传的 X-Forwarded-For / X-Real-IP 请求头。
    默认 False：容器直接暴露公网端口时严禁信任客户端可伪造的转发头，防止绕过限流；
    仅当前置了受信任的 Nginx / SLB 并覆盖转发头时，通过 TRUST_PROXY_HEADERS=true 显式开启。
    """
    return os.getenv("TRUST_PROXY_HEADERS", "false").strip().lower() in ("1", "true", "yes")


def get_client_ip(request: Request) -> str:
    """
    解析请求的真实客户端 IP：
    1. 仅当 TRUST_PROXY_HEADERS=true 时，优先读取受信任反代透传的 X-Forwarded-For（首跳）与 X-Real-IP；
    2. 默认（直连公网）忽略可伪造的转发头，直接读取 TCP 对端地址 request.client.host；
    3. 若均不可用，兜底返回 '127.0.0.1'。
    """
    if _is_proxy_trusted():
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            for part in x_forwarded_for.split(","):
                cleaned = part.strip()
                if cleaned:
                    return cleaned

        x_real_ip = request.headers.get("X-Real-IP")
        if x_real_ip and x_real_ip.strip():
            return x_real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


def _has_valid_demo_bypass(request: Request) -> bool:
    """可选的演示/面试官通行口令校验（配置 DEMO_BYPASS_TOKEN 且请求头匹配时放行日配额）。"""
    expected = os.getenv("DEMO_BYPASS_TOKEN", "").strip()
    if not expected:
        return False
    provided = request.headers.get("X-Demo-Bypass-Token", "").strip()
    return provided == expected


def check_chat_rate_limit(request: Request) -> None:
    """
    挂载在 POST /api/chat 与 /api/chat/stream 的单 IP 频次限流依赖项。
    在执行任何 Agent 思考或大模型 API 调用前进行防刷拦截。
    """
    if _has_valid_demo_bypass(request):
        return
    client_ip = get_client_ip(request)
    _chat_limiter.check(client_ip)


def check_daily_quota_limit(request: Request) -> None:
    """
    挂载在 POST /api/chat 与 /api/chat/stream 的每日双层配额熔断依赖项（单 IP + 全局）。
    按北京时间每日 00:00 自动重置，超出上限直接返回 HTTP 429。
    """
    if _has_valid_demo_bypass(request):
        return
    client_ip = get_client_ip(request)
    _daily_quota_limiter.check_and_increment(client_ip)


def check_api_read_rate_limit(request: Request) -> None:
    """挂载在 /api/knowledge/search、/api/orders、GET /api/tickets 的轻量只读 IP 限流依赖项。"""
    client_ip = get_client_ip(request)
    _api_read_limiter.check(client_ip)


def check_api_write_rate_limit(request: Request) -> None:
    """挂载在 POST/PATCH /api/tickets* 的写操作 IP 限流依赖项。"""
    client_ip = get_client_ip(request)
    _api_write_limiter.check(client_ip)


# 演示重置独立冷却限流器：未配置 DEMO_ADMIN_TOKEN 时，单 IP 每 5 分钟最多重置 1 次
_demo_reset_limiter = SlidingWindowLimiter(
    window_seconds=300,
    default_limit=1,
    env_var="DEMO_RESET_RATE_LIMIT_PER_5MIN",
)


def _get_reset_cooldown_seconds() -> int:
    raw = os.getenv("DEMO_RESET_COOLDOWN_SECONDS")
    if raw:
        try:
            parsed = int(raw.strip())
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return 300


def check_demo_admin_token(request: Request) -> None:
    """
    演示数据重置双层保护：
      1. 若配置了 DEMO_ADMIN_TOKEN：强制校验请求头 X-Demo-Admin-Token，不匹配返回 HTTP 403；
      2. 若未配置 DEMO_ADMIN_TOKEN：启用独立 IP 冷却（默认每 5 分钟最多重置 1 次），超限返回 HTTP 429，
         防止公网演示数据被恶意高频刷乱。
    """
    expected = os.getenv("DEMO_ADMIN_TOKEN", "").strip()
    if expected:
        provided = request.headers.get("X-Demo-Admin-Token", "").strip()
        if provided != expected:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="当前环境已启用演示管理口令保护，请在 X-Demo-Admin-Token 中提供正确的口令。",
            )
        return

    _demo_reset_limiter.window_seconds = _get_reset_cooldown_seconds()
    client_ip = get_client_ip(request)
    try:
        _demo_reset_limiter.check(client_ip)
    except HTTPException as exc:
        retry_after = (exc.headers or {}).get("Retry-After", str(_demo_reset_limiter.window_seconds))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"公开演示环境限制每 {_demo_reset_limiter.window_seconds // 60 or 1} 分钟最多重置 1 次，请在 {retry_after} 秒后重试。",
            headers={"Retry-After": retry_after},
        ) from exc
