# 封装意图类型

from enum import Enum

class Intentions(Enum):
    GREETING = "greeting"
    SEARCH = "search_menu" # 查询菜品
    PROCESS_ORDER = "process_order" # 處理訂單
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

    @staticmethod
    def from_str(s: str) -> 'Intentions':
        """
        将字符串转换为 Intentions 枚举值
        
        Args:
            s: 输入字符串，可以是枚举的值（如 "greeting"）或名称（如 "GREETING"）
        
        Returns:
            对应的 Intentions 枚举值，如果找不到则返回 UNKNOWN
        """
        if not s:
            return Intentions.UNKNOWN
            
        # 首先尝试按值匹配
        for member in Intentions:
            if member.value == s.lower():
                return member
        
        # 然后尝试按名称匹配（大小写不敏感）
        for member in Intentions:
            if member.name.lower() == s.lower():
                return member
        
        # 如果都找不到，返回 UNKNOWN
        return Intentions.UNKNOWN
