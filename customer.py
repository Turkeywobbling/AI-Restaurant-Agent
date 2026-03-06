# 封装的用户类

class Customer:
    def __init__(self, phone):
        self.phone = phone # 当作id使用
        self.orders = None

    def set_orders(self, orders):
        self.orders = orders

    # 后续历史订单查询等功能可以在这里添加        