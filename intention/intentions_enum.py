# 封装意图类型

from enum import Enum

class Intentions(Enum):
    GREETING = "greeting"
    ORDER = "order"
    SEARCH = "search"
    QUERY_ORDER = "query_order" # 查询订单状态
    MODIFY_ORDER = "modify_order" # 修改订单
    CONFIRM_ORDER = "confirm_order" # 确认订单
    CONFIRM_PRICE = "confirm_price" # 确认价格
    CONFIRM_ADDRESS = "confirm_address" # 确认地址
    CONFIRM_PHONE = "confirm_phone" # 确认电话号码
    PLACE_ORDER = "place_order" # 下单
    FAREWELL = "farewell"
    HELP = "help"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown" # 未知意图
