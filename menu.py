class Menu:
    """简单的菜单类，用于快速查找菜品信息"""
    
    def __init__(self, menu_data):
        """
        初始化菜单
        
        Args:
            menu_data: 可以是文件路径、字典或列表
        """
        self.dishes = []
        self.dish_dict = {}  # 名称 -> 菜品信息
        self.price_dict = {}  # 名称 -> 价格
        self.category_dict = {}  # 分类 -> 菜品列表
        
        self._load_menu(menu_data)
        
    def _load_menu(self, menu_data):
        """加载菜单数据"""
        # 如果是文件路径
        if isinstance(menu_data, str):
            import json
            try:
                with open(menu_data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.dishes = data.get('dishes', [])
            except:
                print(f"无法加载文件: {menu_data}")
                self.dishes = []
        
        # 如果是字典
        elif isinstance(menu_data, dict):
            self.dishes = menu_data.get('dishes', [])
        
        # 如果是列表
        elif isinstance(menu_data, list):
            self.dishes = menu_data
        
        # 构建索引
        self._build_index()
    
    def _build_index(self):
        """构建快速查找索引"""
        for dish in self.dishes:
            name = dish.get('name', '')
            if name:
                self.dish_dict[name] = dish
                self.price_dict[name] = dish.get('price', 0)
                
                # 按分类索引
                category = dish.get('category', '其他')
                if category not in self.category_dict:
                    self.category_dict[category] = []
                self.category_dict[category].append(dish)
    
    # ============ 查找方法 ============
    
    def get_dish(self, name: str) -> dict:
        """根据名称获取菜品信息"""
        return self.dish_dict.get(name)
    
    def get_price(self, name: str) -> float:
        """获取菜品价格"""
        return self.price_dict.get(name, 0)
    
    def get_by_category(self, category: str) -> list:
        """获取某分类的所有菜品"""
        return self.category_dict.get(category, [])
    
    def search_by_name(self, keyword: str) -> list:
        """根据关键词搜索菜品"""
        results = []
        for name, dish in self.dish_dict.items():
            if keyword in name:
                results.append(dish)
        return results
    
    def get_all_names(self) -> list:
        """获取所有菜品名称"""
        return list(self.dish_dict.keys())
    
    def get_all_categories(self) -> list:
        """获取所有分类"""
        return list(self.category_dict.keys())
    
    # ============ 检查方法 ============
    
    def exists(self, name: str) -> bool:
        """检查菜品是否存在"""
        return name in self.dish_dict
    
    def is_spicy(self, name: str) -> bool:
        """检查是否是辣菜"""
        dish = self.get_dish(name)
        if dish:
            spicy = dish.get('spicy_level', '不辣')
            return spicy != '不辣'
        return False
    
    # ============ 统计方法 ============
    
    def count(self) -> int:
        """获取菜品总数"""
        return len(self.dishes)
    
    def get_price_range(self) -> tuple:
        """获取价格范围"""
        prices = [d.get('price', 0) for d in self.dishes]
        if prices:
            return (min(prices), max(prices))
        return (0, 0)
    
    # ============ 显示方法 ============
    
    def show_all(self):
        """显示所有菜品"""
        print("\n📋 完整菜单:")
        print("-" * 60)
        for i, dish in enumerate(self.dishes, 1):
            name = dish.get('name', '')
            price = dish.get('price', 0)
            spicy = dish.get('spicy_level', '不辣')
            print(f"{i:2}. {name:15} {price:4}元 [{spicy}]")
    
    def show_category(self, category: str):
        """显示某分类的菜品"""
        dishes = self.get_by_category(category)
        if not dishes:
            print(f"没有找到分类: {category}")
            return
        
        print(f"\n📋 {category} 菜品:")
        print("-" * 40)
        for dish in dishes:
            name = dish.get('name', '')
            price = dish.get('price', 0)
            print(f"   • {name} - {price}元")
    
    def __getitem__(self, name: str) -> dict:
        """支持 menu['麻婆豆腐'] 语法"""
        return self.get_dish(name)
    
    def __contains__(self, name: str) -> bool:
        """支持 '麻婆豆腐' in menu 语法"""
        return self.exists(name)
    
    def __len__(self) -> int:
        """支持 len(menu)"""
        return self.count()
    
    def __str__(self) -> str:
        """打印菜单信息"""
        return f"Menu(共 {self.count()} 个菜品, {len(self.category_dict)} 个分类)"

