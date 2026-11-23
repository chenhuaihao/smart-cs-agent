"""
app/db/seed.py

电商订单与售后工单精选种子数据模块（v2.0 P4-A 精简加固）：
  - 24 条干净、无调试字母后缀的标准订单，完整覆盖售后全生命周期边界场景：
    待发货（48h 内 / 超 48h）、运输中（正常 / 停滞异常）、派送中、
    已签收（7天内 / 7~15天换新期 / 超15天质保内 / 过保 / 配件在保与过保 / 刻字定制）、
    已退货、退款中、已取消；
  - 3 条覆盖「待处理 / 处理中 / 已完成」全状态的精选演示工单；
  - 提供 reset_demo_data() 一键清空并重建干净的标准订单 + 工单，防止演示环境脏数据累积。
"""

import logging
from typing import List, Optional, Tuple

from app.db.database import get_db_connection, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 24 条覆盖完整售后边界的精选标准订单（纯数字单号，以 2026-11-23 为基准日期）
# 结构: (order_id, user_name, product_name, amount, status, tracking_no, created_at)
# ---------------------------------------------------------------------------
MOCK_ORDERS: List[Tuple[str, str, str, float, str, Optional[str], str]] = [
    # 1. 基准黄金单：已签收超 15 天，在 1 年官方质保期内
    ("ORD20261101", "张伟", "智能降噪无线蓝牙耳机 Pro", 399.00, "已签收", "SF1428570001", "2026-11-01 10:15:00"),
    # 2. 机械键盘：已签收超 15 天，用于硬件连击/失灵自助排查与保修
    ("ORD20261102", "李娜", "机械键盘 87键 红轴 RGB背光", 269.00, "已签收", "ZT8823901234", "2026-11-02 14:30:00"),
    # 3. 已退货完成订单（无理由退货闭环）
    ("ORD20261103", "赵敏", "骨传导运动蓝牙耳机 S3", 499.00, "已退货", "SF1428570003", "2026-11-03 09:40:00"),
    # 4. 大件家居数码：已签收，质保期内
    ("ORD20261105", "陈敏", "人体工学电脑椅 黑色尼龙脚", 899.00, "已签收", "DBL3344556677", "2026-11-05 11:10:00"),
    # 5. 已退货完成订单（质量瑕疵退货闭环）
    ("ORD20261106", "黄磊", "十合一 Type-C 多功能扩展坞", 189.00, "已退货", "JD0099881106", "2026-11-06 16:20:00"),
    # 6. 已签收 15 天临界单（质保期内）
    ("ORD20261108", "徐静", "2K 165Hz IPS 电竞显示器 27英寸", 1099.00, "已签收", "JD0099881108", "2026-11-08 13:15:00"),
    # 7. 已取消订单
    ("ORD20261110", "吴磊", "筋膜枪 便携肌肉放松仪", 199.00, "已取消", None, "2026-11-10 17:25:00"),
    # 8. 已签收 11 天：超过 7 天无理由，处于 15 天质量问题换新期内
    ("ORD20261112", "林峰", "头戴式降噪耳机 Max", 799.00, "已签收", "JD0099881112", "2026-11-12 11:30:00"),
    # 9. 已签收 9 天：处于 15 天质量问题换新期内
    ("ORD20261114", "高鹏", "三模无线客制化机械键盘 75配列", 529.00, "已签收", "SF1428570014", "2026-11-14 19:05:00"),
    # 10. 物流停滞异常：运输中且发货超 72 小时无更新
    ("ORD20261115", "沈洁", "羊绒围巾 冬季保暖纯色 灰色", 359.00, "运输中", "SF1428570015", "2026-11-15 10:00:00"),
    # 11. 已签收 7 天内：适用 7 天价保与发票重开咨询
    ("ORD20261116", "郭涛", "100W 桌面多口氮化镓充电站", 219.00, "已签收", "ZT8823901116", "2026-11-16 15:45:00"),
    # 12. 退款中订单（退货寄回验收审核中）
    ("ORD20261117", "胡艳", "家用无雾加湿器 4L 静音款", 219.00, "退款中", "YTO5566771117", "2026-11-17 08:50:00"),
    # 13. 待发货超时：下单已超 48 小时未发货（可催发货/取消订单）
    ("ORD20261118", "何平", "复古电水壶 304不锈钢 1.7L", 119.00, "待发货", None, "2026-11-18 09:00:00"),
    # 14. 专属个性化镭射刻字定制耳机（不支持 7 天无理由退货，质量问题可换新/保修）
    ("ORD20261119", "马超", "智能降噪蓝牙耳机【专属个性化镭射刻字版】", 459.00, "已签收", "SF1428570019", "2026-11-19 12:00:00"),
    # 15. 正常在途运输中
    ("ORD20261120", "王强", "超轻羽绒服 男款 黑色 L码", 599.00, "运输中", "YTO5566778899", "2026-11-20 09:20:00"),
    # 16. 签收 2 天内：7 天无理由退货黄金演示单
    ("ORD20261121", "陈萌", "真无线蓝牙耳机 Air", 299.00, "已签收", "SF1428570021", "2026-11-21 14:00:00"),
    # 17. 待发货：刚下单处于正常 48 小时发货时效内
    ("ORD20261122", "郑秀华", "智能保温杯 测温显温款 500ml", 89.00, "待发货", None, "2026-11-22 10:00:00"),
    # 18. 派送中订单（即将送达）
    ("ORD20261123", "朱晨", "无线双模电竞鼠标 8000Hz", 299.00, "派送中", "SF1428570023", "2026-11-21 18:30:00"),
    # 19. 待发货（满减凑单/优惠券退款咨询样例）
    ("ORD20261124", "杨欣", "磁吸无线充电宝 10000mAh + 保护壳套装", 318.00, "待发货", None, "2026-11-23 09:10:00"),
    # 20. 已签收 45 天：超过 15 天换新期，在 1 年官方质保期内
    ("ORD20261008", "孙浩", "4K 144Hz 电竞显示器 27英寸", 1399.00, "已签收", "JD0099887766", "2026-10-08 15:20:00"),
    # 21. 特殊品类配件：原装数据线（3 个月质保），购买不足 1 个月在保
    ("ORD20261025", "许杰", "原装磁吸快充数据线 1.2米", 39.00, "已签收", "SF1428570025", "2026-10-25 11:00:00"),
    # 22. 特殊品类配件：氮化镓快充头+数据线（3 个月质保），购买超 4 个月已过保
    ("ORD20260715", "刘洋", "65W氮化镓快充插头+编织数据线套装", 89.00, "已签收", "ZT8823907715", "2026-07-15 16:30:00"),
    # 23. 购买超 1 年：已过官方 1 年质保期（适用质保期外付费维修）
    ("ORD20250510", "周芳", "多功能电火锅 6L大容量", 189.00, "已签收", "ZT8823905678", "2025-05-10 13:40:00"),
    # 24. 购买超 1 年：电子产品过保样例
    ("ORD20250820", "钱明", "智能运动手环 8代 NFC版", 249.00, "已签收", "SF1428570820", "2025-08-20 20:15:00"),
]

# 保持 ALL_ORDERS 导出别名以兼容可能引用它的脚本
ALL_ORDERS: List[Tuple[str, str, str, float, str, Optional[str], str]] = list(MOCK_ORDERS)


def seed_orders() -> int:
    """
    幂等初始化 24 条精选标准演示订单：
    仅当某 order_id 不存在时插入，已存在记录原样保留。
    """
    init_db()

    check_sql = "SELECT 1 FROM orders WHERE order_id = ?;"
    insert_sql = """
    INSERT INTO orders (
        order_id, user_name, product_name, amount, status, tracking_no, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?);
    """

    new_inserted = 0
    with get_db_connection() as conn:
        cursor = conn.cursor()
        for order in ALL_ORDERS:
            cursor.execute(check_sql, (order[0],))
            if not cursor.fetchone():
                cursor.execute(insert_sql, order)
                new_inserted += 1
        conn.commit()

    if new_inserted > 0:
        logger.info("演示订单初始化完成：本次新增写入 %d 条，标准底库共 %d 条。", new_inserted, len(ALL_ORDERS))
    else:
        logger.info("演示订单检查完成：全部 %d 条标准订单均已就绪（幂等无新增）。", len(ALL_ORDERS))

    return new_inserted


# ---------------------------------------------------------------------------
# 3 条覆盖「待处理 / 处理中 / 已完成」全状态的精选演示工单
# 结构: (ticket_id, order_id, issue_type, reason, status, summary, created_at)
# ---------------------------------------------------------------------------
MOCK_TICKETS: List[Tuple[str, str, str, str, str, str, str]] = [
    (
        "TKT20261120-DEMO01",
        "ORD20261101",
        "退货",
        "耳机佩戴夹头且耳罩偏硬，申请售后评估退货方案",
        "待处理",
        "买家反馈佩戴不适申请退货，待客服审核",
        "2026-11-20 14:20:00",
    ),
    (
        "TKT20261118-DEMO02",
        "ORD20261112",
        "换货",
        "耳机左耳出现断续杂音，处于15天质量问题换新期内",
        "处理中",
        "售后检测已受理，正在安排仓库寄发换新包裹",
        "2026-11-18 10:30:00",
    ),
    (
        "TKT20261115-DEMO03",
        "ORD20261008",
        "售后咨询",
        "咨询4K电竞显示器DP1.4接口与固件升级方法",
        "已完成",
        "已向用户发送固件包下载链接与操作指引，用户已确认解决",
        "2026-11-15 16:45:00",
    ),
]


def seed_tickets() -> int:
    """
    幂等初始化工单演示数据：
    若 tickets 表为空，则自动补录 3 条状态分别为「待处理、处理中、已完成」的示例工单；
    若已有工单数据则跳过。
    """
    init_db()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS cnt FROM tickets;")
        count = cursor.fetchone()["cnt"]
        if count > 0:
            logger.info("tickets 表中已有 %d 条工单，跳过演示数据补录。", count)
            return 0

        insert_sql = """
        INSERT INTO tickets (ticket_id, order_id, issue_type, reason, status, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        cursor.executemany(insert_sql, MOCK_TICKETS)
        conn.commit()

    logger.info("成功初始化 3 条多状态演示工单（待处理/处理中/已完成）。")
    return len(MOCK_TICKETS)


def reset_tickets() -> int:
    """
    一键重置工单表为 3 条干净的初始演示工单（保留向后兼容签名）。
    """
    _, ticket_count = reset_demo_data()
    return ticket_count


def reset_demo_data() -> Tuple[int, int]:
    """
    一键清空并重建标准演示订单（24 条）与演示工单（3 条），
    彻底清除历史脏数据或旧版 500 条冗余订单，防止演示环境数据膨胀。
    """
    init_db()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        # 先清子表 tickets 再清父表 orders，避免外键约束冲突
        cursor.execute("DELETE FROM tickets;")
        cursor.execute("DELETE FROM orders;")
        cursor.executemany(
            """
            INSERT INTO orders (
                order_id, user_name, product_name, amount, status, tracking_no, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            ALL_ORDERS,
        )
        cursor.executemany(
            """
            INSERT INTO tickets (ticket_id, order_id, issue_type, reason, status, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            MOCK_TICKETS,
        )
        conn.commit()

    logger.info(
        "演示数据重置完成：已重建 %d 条标准订单与 %d 条演示工单。",
        len(ALL_ORDERS),
        len(MOCK_TICKETS),
    )
    return len(ALL_ORDERS), len(MOCK_TICKETS)


if __name__ == "__main__":
    seed_orders()
    seed_tickets()
