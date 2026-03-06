# 封装阶段
from enum import Enum

class stage(Enum):
    ORDERING = "ordering" # 点餐阶段
    CONFIRMING_PRICE = "confirming_price" # 确认价格阶段
    CONFIRMING_ADDRESS = "confirming_address" # 确认地址阶段
    PLACING_ORDER = "placing_order" # 下单阶段
    COMPLETED = "completed" # 订单完成阶段