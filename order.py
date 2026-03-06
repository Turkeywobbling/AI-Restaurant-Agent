import json

class order_manager:
    def __init__(self, id):
        self.order_data = {}

        self.order_data["id"] = id    
        self.order_data["customer"] = None
        self.order_data["items"] = {}
        self.order_data["total_price"] = 0
        self.order_data["delivery_location"] = None

    def add_item(self, item, price, quantity=1):
        if item in self.order_data["items"]:
            self.order_data["items"][item] += quantity
        else:
            self.order_data["items"][item] = quantity
        
        self.order_data["total_price"] += price * quantity

    def set_customer(self, customer):
        self.order_data["customer"] = customer

    def set_delivery_location(self, location):
        self.order_data["delivery_location"] = location

    def change_item(self, old_item, new_item, old_price, new_price):
        if old_item in self.order_data["items"]:
            quantity = self.order_data["items"].pop(old_item)
            self.order_data["items"][new_item] = quantity
            self.order_data["total_price"] += (new_price - old_price) * quantity

    def remove_item(self, item, price, quantity=1):
        if item in self.order_data["items"]:
            self.order_data["total_price"] -= price * quantity  

    def get_order_summary(self):              
        json_summary = json.dumps(self.order_data, indent=4, ensure_ascii=False)
        return json_summary

    # 订单增删改查