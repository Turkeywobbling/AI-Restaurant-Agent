import re
import jieba
import jieba.posseg as pseg
from typing import Dict, List, Optional, Set
import os
import json

from log.log_utils import log_utils

class NLPEntityExtractor:
    """使用 NLP 的实体提取器"""
    
    def __init__(self, menus_path):
        """
        初始化 NLP 实体提取器
        
        Args:
            menu_items: 菜单菜品列表
        """
        self.data_path = menus_path
        self.menu_items = self._load_menu()
        self.id_to_item = {item['id']: item for item in self.menu_items}
        self.load_dish()
        self.dish_name_set = set(self.dish_names)
        
        # 构建菜品名索引（用于快速查找）
        self.dish_index = self._build_dish_index()
        
        # 加载自定义词典（把菜品名加入 jieba 词典，提高分词准确率）
        self._load_dish_to_jieba()
        
        # 点餐相关的动词和量词
        self.order_verbs = {'点', '要', '来', '叫', '买', '吃', '下单'}
        self.quantifiers = {'份', '个', '盘', '碗', '碗', '碟'}
        
        log_utils.d(f"📚 NLP实体提取器初始化完成")
        log_utils.d(f"   📊 菜品数量: {len(self.dish_names)}")
        log_utils.d(f"   📝 分词词典已更新")

    def _load_menu(self) -> List[Dict]:
        """加载菜单数据"""
        if not os.path.exists(self.data_path):
            log_utils.d(f"  ⚠️ 菜单文件不存在，创建示例菜单...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            menu = json.load(f)
        return menu["dishes"]

    def load_dish(self) -> List:
        self.dish_names = []
        for dish in self.menu_items:
            self.dish_names.append(dish["name"])
     
    
    def _build_dish_index(self) -> Dict:
        """构建菜品名索引"""
        index = {}
        for dish in self.dish_names:
            # 按长度排序，优先匹配长词
            index[dish] = len(dish)
        return index
    
    def _load_dish_to_jieba(self):
        """将菜品名加入 jieba 词典"""
        for dish in self.dish_names:
            jieba.add_word(dish)
            # 也可以添加菜品名的变体
            if '麻辣' in dish:
                jieba.add_word('麻辣')
    

    def extract_dish_nlp(self, text: str) -> Optional[str]:
        """
        使用 NLP 方法提取菜品名
        
        Args:
            text: 用户输入文本
            
        Returns:
            提取到的菜品名，如果没有返回 None
        """
        log_utils.d(f"\n🔍 NLP提取从: '{text}'")
        
        # 1. 先检查是否直接包含完整菜品名
        for dish_name in sorted(self.dish_names, key=len, reverse=True):
            if dish_name in text:
                log_utils.d(f"   ✅ 直接匹配: {dish_name}")
                return dish_name
        
        # 2. 使用 jieba 分词 + 词性标注
        words = pseg.cut(text)
        
        candidates = []
        for word, flag in words:
            log_utils.d(f"   词: {word} ({flag})")
            
            # 名词、动名词、专有名词 可能是菜品
            if flag.startswith('n') or flag in ['vn', 'an']:
                candidates.append(word)
            
            # 如果是动词且是点餐相关，后面跟着的词可能是菜品
            if flag.startswith('v') and word in self.order_verbs:
                # 记录这个动词，后面会检查
                pass
        
        # 3. 检查候选词是否在菜单中
        for candidate in candidates:
            # 精确匹配
            if candidate in self.dish_name_set:
                log_utils.d(f"   ✅ 分词匹配: {candidate}")
                return candidate
            
            # 模糊匹配：检查是否是菜品名的子串
            for dish_name in self.dish_names:
                if candidate in dish_name and len(candidate) >= 2:
                    log_utils.d(f"   ✅ 子串匹配: {candidate} -> {dish_name}")
                    return dish_name
        
        # 4. 提取所有可能的中文词组
        words = re.findall(r'[\u4e00-\u9fa5]{2,}', text)
        for word in words:
            if word in self.dish_name_set:
                log_utils.d(f"   ✅ 关键词匹配: {word}")
                return word
        
        return None
    
    def extract_dish_ngram(self, text: str, max_n: int = 3) -> Optional[str]:
        """
        使用 N-gram 方法提取菜品名
        
        Args:
            text: 用户输入文本
            max_n: 最大词组长度
            
        Returns:
            提取到的菜品名
        """
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5]', '', text)  # 只保留中文
        
        # 生成所有可能的 N-gram
        for n in range(max_n, 1, -1):  # 从长到短
            for i in range(len(text) - n + 1):
                candidate = text[i:i+n]
                if candidate in self.dish_name_set:
                    log_utils.d(f"   ✅ N-gram匹配 ({n}字): {candidate}")
                    return candidate
        
        return None
    
    def extract_all_methods(self, text: str) -> Dict:
        """
        使用多种方法提取，返回最佳结果
        
        Returns:
            {
                'dish': 提取到的菜品,
                'method': 使用的方法,
                'confidence': 置信度
            }
        """
        result = {
            'dish': None,
            'method': None,
            'confidence': 0.0
        }
        
        # 方法1：直接匹配（最高置信度）
        for dish_name in sorted(self.dish_names, key=len, reverse=True):
            if dish_name in text:
                result.update({
                    'dish': dish_name,
                    'method': 'direct_match',
                    'confidence': 1.0
                })
                return result
        
        # 方法2：NLP分词匹配
        nlp_result = self.extract_dish_nlp(text)
        if nlp_result:
            result.update({
                'dish': nlp_result,
                'method': 'nlp_match',
                'confidence': 0.9
            })
            return result
        
        # 方法3：N-gram匹配
        ngram_result = self.extract_dish_ngram(text)
        if ngram_result:
            result.update({
                'dish': ngram_result,
                'method': 'ngram_match',
                'confidence': 0.8
            })
            return result
        
        return result
    
    def extract(self, text: str) -> Dict:
        """
        完整实体提取（包括数量和价格等）
        
        Args:
            text: 用户输入
            
        Returns:
            包含所有实体的字典
        """
        # 提取菜品（使用多种方法）
        dish_result = self.extract_all_methods(text)
        
        # 提取数量和价格
        quantity = 1
        price = None
        
        # 提取数量
        quantity_match = re.search(r'(\d+)\s*份', text)
        if quantity_match:
            quantity = int(quantity_match.group(1))
        
        # 提取价格
        price_match = re.search(r'(\d+)\s*元', text)
        if price_match:
            price = float(price_match.group(1))
        
        return {
            'dish': dish_result['dish'],
            'quantity': quantity,
            'price': price,
            'phone': None,
            'address': None,
            '_debug': {
                'method': dish_result['method'],
                'confidence': dish_result['confidence']
            }
        }

