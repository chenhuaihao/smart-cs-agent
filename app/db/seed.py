"""
app/db/seed.py

电商订单模拟种子数据模块。
扩充至 500 条高质量订单数据：
- 严格保留 15 条覆盖售后核心边界场景的手工精选测试单；
- 固定 random.seed(42) 确定性批量生成 485 条多品类、全物流状态真实感订单；
- 保持按 order_id 幂等补录，已存在跳过，平滑升级不覆盖原数据。
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from app.db.database import get_db_connection, init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 15 条覆盖完整售后测试边界的精选中文电商订单（以 2026-11 为基准时间）
# 结构: (order_id, user_name, product_name, amount, status, tracking_no, created_at)
# ---------------------------------------------------------------------------
MOCK_ORDERS: List[Tuple[str, str, str, float, str, Optional[str], str]] = [
    # 1. 存量基准黄金测试单（保留）：签收超过 15 天但在 1 年质保内
    ("ORD20261101", "张伟", "智能降噪无线蓝牙耳机 Pro", 399.00, "已签收", "SF1428570001", "2026-11-01 10:15:00"),

    # 2. 签收 2 天内（7 天无理由退货期内）
    ("ORD20261121A", "陈萌", "真无线蓝牙耳机 Air", 299.00, "已签收", "SF1428570021", "2026-11-21 14:00:00"),

    # 3. 签收 10 天（超过 7 天无理由，但在 15 天质量问题换新期内）
    ("ORD20261112H", "林峰", "头戴式降噪耳机 Max", 799.00, "已签收", "JD0099881112", "2026-11-12 11:30:00"),

    # 4. 签收 45 天（超过 15 天换新期，但在 1 年官方质保期内）
    ("ORD20261008", "孙浩", "4K 144Hz 电竞显示器 27英寸", 1399.00, "已签收", "JD0099887766", "2026-10-08 15:20:00"),

    # 5. 购买 1 年半前（超过 1 年质保期，已过保）
    ("ORD20250510", "周芳", "多功能电火锅 6L大容量", 189.00, "已签收", "ZT8823905678", "2025-05-10 13:40:00"),

    # 6. 待发货：下单已超 48 小时未发货（超时未发/买家催发货/支持取消拦截）
    ("ORD20261118D", "何平", "复古电水壶 304不锈钢 1.7L", 119.00, "待发货", None, "2026-11-18 09:00:00"),

    # 7. 待发货：刚下单处于正常 48 小时发货时效内
    ("ORD20261122N", "郑秀华", "智能保温杯 测温显温款 500ml", 89.00, "待发货", None, "2026-11-22 10:00:00"),

    # 8. 正常在途运输中
    ("ORD20261121T", "王强", "超轻羽绒服 男款 黑色 L码", 599.00, "运输中", "YTO5566778899", "2026-11-21 09:20:00"),

    # 9. 物流停滞异常：发货后超过 72 小时无更新
    ("ORD20261115S", "沈洁", "羊绒围巾 冬季保暖纯色 灰色", 359.00, "运输中", "SF1428570015", "2026-11-15 10:00:00"),

    # 10. 已取消订单
    ("ORD20261110", "吴磊", "筋膜枪 便携肌肉放松仪", 199.00, "已取消", None, "2026-11-10 17:25:00"),

    # 11. 特殊品类：充电头/数据线易损配件（质保期 3 个月），已购买 4 个月已过保
    ("ORD20260715", "刘洋", "65W氮化镓快充插头+编织数据线套装", 89.00, "已签收", "ZT8823907715", "2026-07-15 16:30:00"),

    # 12. 特殊品类：原装快充数据线配件（质保期 3 个月），购买不到 1 个月在保
    ("ORD20261025", "许杰", "原装磁吸快充数据线 1.2米", 39.00, "已签收", "SF1428570025", "2026-10-25 11:00:00"),

    # 13. 特殊品类：机械键盘（用于硬件常见故障自助排查与连击测试）
    ("ORD20261102", "李娜", "机械键盘 87键 红轴 RGB背光", 269.00, "已签收", "ZT8823901234", "2026-11-02 14:30:00"),

    # 14. 特殊品类：专属个性化镭射刻字版耳机（不支持 7 天无理由退货，仅保修）
    ("ORD20261119C", "马超", "智能降噪蓝牙耳机【专属个性化镭射刻字版】", 459.00, "已签收", "SF1428570019", "2026-11-19 12:00:00"),

    # 15. 大件家居数码常规备用订单
    ("ORD20261105", "陈敏", "人体工学电脑椅 黑色尼龙脚", 899.00, "已签收", "DBL3344556677", "2026-11-05 11:10:00"),
]


def generate_mock_orders(target_count: int = 500) -> List[Tuple[str, str, str, float, str, Optional[str], str]]:
    """
    固定种子确定性生成模拟订单，平滑补齐至 target_count 条（默认 500 条）。
    前 15 条为人工精选边界订单，后 485 条为真实感模拟订单。
    覆盖 2025-01 至 2026-11 时间跨度、多品类与全物流状态。
    """
    orders = list(MOCK_ORDERS)
    needed = target_count - len(orders)
    if needed <= 0:
        return orders

    rng = random.Random(42)

    # 常见姓氏与名字库
    surnames = ["王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴", "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "郑"]
    names = ["伟", "芳", "秀英", "娜", "敏", "静", "丽", "强", "磊", "军", "洋", "勇", "艳", "杰", "娟", "涛", "明", "超", "秀兰", "浩", "平", "洁", "鹏", "欣", "晨"]

    # 商品库（覆盖 3C 数码、生活家电、居家日用、服饰箱包、定制商品等，含合理基准单价）
    products = [
        ("真无线降噪蓝牙耳机", 299.00),
        ("头戴式无线降噪耳机", 699.00),
        ("骨传导运动蓝牙耳机", 499.00),
        ("机械键盘 87键 红轴", 269.00),
        ("机械键盘 104键 茶轴 RGB", 399.00),
        ("三模无线客制化机械键盘", 529.00),
        ("4K 144Hz 电竞显示器 27英寸", 1499.00),
        ("2K 165Hz IPS小金刚显示器", 1099.00),
        ("无线双模电竞鼠标 8000Hz", 299.00),
        ("人体工学垂直办公无线鼠标", 159.00),
        ("65W氮化镓快充插头+数据线", 89.00),
        ("100W桌面多口氮化镓充电站", 219.00),
        ("十合一Type-C多功能扩展坞", 189.00),
        ("磁吸无线充电宝 10000mAh", 139.00),
        ("大容量快充移动电源 20000mAh", 169.00),
        ("智能运动手环 8代", 199.00),
        ("多功能电火锅 6L大容量", 189.00),
        ("复古电水壶 304不锈钢 1.7L", 119.00),
        ("智能恒温电热水壶 1.5L", 159.00),
        ("空气炸锅 可视化大容量 5L", 259.00),
        ("低音破壁料理机 1.75L", 329.00),
        ("无线手持吸尘器 大吸力款", 699.00),
        ("家用无雾加湿器 4L", 219.00),
        ("挂烫机 手持便携式折叠款", 129.00),
        ("人体工学电脑椅 黑色尼龙脚", 899.00),
        ("智能保温杯 测温显温款 500ml", 89.00),
        ("超轻羽绒服 男款 黑色 L码", 599.00),
        ("中长款极寒羽绒服 女款 白鸭绒", 799.00),
        ("羊绒围巾 冬季保暖纯色 灰色", 359.00),
        ("商务防泼水双肩电脑包", 169.00),
        ("筋膜枪 便携肌肉放松仪", 199.00),
        ("智能降噪蓝牙耳机【专属个性化镭射刻字版】", 459.00),
        ("客制化机械键盘【个人ID激光刻印版】", 599.00),
        ("原装磁吸快充数据线 1.2米", 39.00),
    ]

    # 物流状态分布与对应单号前缀
    status_weights = [("已签收", 0.70), ("运输中", 0.14), ("待发货", 0.11), ("已取消", 0.05)]
    couriers = ["SF", "JD", "ZT", "YTO", "STO"]

    # 构造 2025-01-01 至 2026-11-20 的时间序列（保证确定性递增）
    start_dt = datetime(2025, 1, 1, 9, 0, 0)
    end_dt = datetime(2026, 11, 20, 18, 0, 0)
    total_seconds = int((end_dt - start_dt).total_seconds())
    step_seconds = total_seconds // needed

    for i in range(needed):
        seq = 16 + i
        cur_dt = start_dt + timedelta(seconds=i * step_seconds + rng.randint(0, min(1800, max(1, step_seconds))))
        dt_str = cur_dt.strftime("%Y%m%d")
        created_at = cur_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 订单号格式：ORD + YYYYMMDD + 4位序号（如 ORD202501050016）
        order_id = f"ORD{dt_str}{seq:04d}"

        user_name = rng.choice(surnames) + rng.choice(names)
        prod_name, base_price = rng.choice(products)
        # 价格允许适度微小波动
        price_jitter = rng.choice([-20, -10, 0, 10, 20, 30])
        amount = round(max(19.0, base_price + price_jitter), 2)

        # 状态按权重随机选择
        rand_val = rng.random()
        cumulative = 0.0
        chosen_status = "已签收"
        for st, weight in status_weights:
            cumulative += weight
            if rand_val <= cumulative:
                chosen_status = st
                break

        # 运单号生成逻辑
        tracking_no: Optional[str] = None
        if chosen_status in ("已签收", "运输中"):
            prefix = rng.choice(couriers)
            tracking_digits = f"{rng.randint(1000000000, 9999999999)}"
            tracking_no = f"{prefix}{tracking_digits}"

        orders.append((order_id, user_name, prod_name, amount, chosen_status, tracking_no, created_at))

    return orders


# 完整 500 条确定性订单数据集
ALL_ORDERS: List[Tuple[str, str, str, float, str, Optional[str], str]] = generate_mock_orders(500)


def seed_orders() -> int:
    """
    向 orders 表幂等灌入 500 条模拟订单数据。

    平滑补录机制：
    1. 确保数据库表已初始化；
    2. 遍历 ALL_ORDERS（15条基准边界单 + 485条确定性生成模拟单），
       通过 SELECT 检查 order_id 是否已存在；
    3. 仅对缺失的 order_id 执行 INSERT，已存在的订单记录原样保留；
    4. 无需删库即可自动补齐新增的边界测试订单。

    Returns:
        int: 本次实际新增写入的订单条数。
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
        logger.info("Mock 订单底库补录完成！本次新增写入 %d 条订单，当前底库总计 500 条。", new_inserted)
    else:
        logger.info("Mock 订单底库检查完成：全部 %d 条订单均已就绪（幂等无新增）。", len(ALL_ORDERS))

    return new_inserted


if __name__ == "__main__":
    seed_orders()
