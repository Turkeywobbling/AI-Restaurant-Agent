import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import pickle
import hashlib

from typing import List, Dict, Optional
from datetime import datetime
import re
import time
import torch

os.environ['HF_HUB_OFFLINE'] = '1'  # 完全离线模式

class FAISSMenuStore:
    """基于FAISS的菜单向量存储"""
    
    def __init__(self, data_path: str = "./data/menu.json", embedding_dim: int = 384):
        """
        初始化FAISS存储
        
        Args:
            data_path: 菜单JSON路径
            embedding_dim: 向量维度（all-MiniLM-L6-v2是384维）
        """
        print("\n🔧 初始化FAISS向量存储...")
        
        self.data_path = data_path
        self.embedding_dim = embedding_dim
        self.index_path = "./data/faiss_index.bin"
        self.metadata_path = "./data/faiss_metadata.pkl"
        
        # 1. 初始化embedding模型
        print("  加载embedding模型...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"  ✅ 模型加载完成，维度: {self.embedder.get_sentence_embedding_dimension()}")
        
        # 2. 初始化或加载FAISS索引
        self._init_index()
        
        # 3. 加载菜单数据
        self.menu_items = self._load_menu()
        self.id_to_item = {item['id']: item for item in self.menu_items}
        
        print(f"  📊 共加载 {len(self.menu_items)} 个菜品")

        # 开始菜品转向量
        print('开始菜品转向量')
        self.build_index()
    
    def _init_index(self):
        """初始化FAISS索引"""
        if os.path.exists(self.index_path):
            # 加载已有索引
            print("  加载已有FAISS索引...")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.id_to_index, self.index_to_id = pickle.load(f)
            print(f"  ✅ 加载完成，索引大小: {self.index.ntotal}")
        else:
            # 创建新索引
            print("  创建新FAISS索引...")
            # 使用L2距离的索引
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.id_to_index = {}  # 菜品ID到索引位置的映射
            self.index_to_id = []  # 索引位置到菜品ID的映射
            print("  ✅ 新索引创建完成")
            return
    
    def _load_menu(self) -> List[Dict]:
        """加载菜单数据"""
        if not os.path.exists(self.data_path):
            print(f"  ⚠️ 菜单文件不存在，创建示例菜单...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            menu = json.load(f)
        return menu["dishes"]
    
    def _get_dish_text(self, dish: Dict) -> str:
        """获取菜品的文本表示（用于向量化）"""
        return f"""
        菜品：{dish['name']}
        价格：{dish['price']}元
        描述：{dish['description']}
        配料：{', '.join(dish['ingredients'])}
        辣度：{dish['spicy_level']}
        类别：{dish['category_name']}
        标签：{', '.join(dish['tags'])}
        """
    
    def build_index(self):
        """构建FAISS索引"""
        print("\n🔨 构建FAISS索引...")
        
        if self.index.ntotal > 0:
            print(f"  索引已存在，大小: {self.index.ntotal}")
            return
        
        # 为每个菜品生成向量
        embeddings = []
        
        for dish in self.menu_items:
            text = self._get_dish_text(dish)
            embedding = self.embedder.encode(text)
            embeddings.append(embedding)
            
            # 记录映射
            idx = len(self.index_to_id)
            self.id_to_index[dish['id']] = idx
            self.index_to_id.append(dish['id'])
        
        # 转换为numpy数组
        embeddings_array = np.array(embeddings).astype('float32')
        
        # 添加到FAISS索引
        self.index.add(embeddings_array)
        
        # 保存索引
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump((self.id_to_index, self.index_to_id), f)
        
        print(f"  ✅ 索引构建完成，共 {self.index.ntotal} 个向量")
        print(f"  💾 索引已保存到 {self.index_path}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        搜索相似菜品
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似菜品列表
        """
        # 生成查询向量
        query_embedding = self.embedder.encode(query).astype('float32').reshape(1, -1)
        
        # 搜索
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS可能返回-1
                continue
            
            dish_id = self.index_to_id[idx]
            dish = self.id_to_item[dish_id]
            
            # 计算相似度分数（距离越小越相似）
            similarity = 1 / (1 + dist)
            
            results.append({
                "rank": i + 1,
                "id": dish_id,
                "name": dish['name'],
                "price": dish['price'],
                "description": dish['description'],
                "spicy_level": dish['spicy_level'],
                "category": dish['category_name'],
                "similarity": round(similarity, 3),
                "distance": float(dist)
            })
        
        return results
    
    def filter_search(self, query: str, filters: Dict, k: int = 10) -> List[Dict]:
        """
        带过滤条件的搜索
        
        Args:
            query: 查询文本
            filters: 过滤条件，如 {"max_price": 40, "category": "川菜"}
            k: 先检索更多结果再过滤
        """
        # 先搜索更多结果
        all_results = self.search(query, k=k*2)
        
        # 应用过滤
        filtered = []
        for dish in all_results:
            match = True
            
            # 价格上限过滤
            if "max_price" in filters and dish['price'] > filters['max_price']:
                match = False
            
            # 价格下限过滤
            if "min_price" in filters and dish['price'] < filters['min_price']:
                match = False
            
            # 类别过滤
            if "category" in filters and dish['category_name'] != filters['category']:
                match = False
            
            # 辣度过滤
            if "spicy_level" in filters and dish['spicy_level'] != filters['spicy_level']:
                match = False
            
            if match:
                filtered.append(dish)
        
        return filtered[:k]
    
    def get_by_category(self, category: str) -> List[Dict]:
        """按类别获取菜品"""
        return [d for d in self.menu_items if d['category'] == category]
    
    def get_by_price_range(self, min_price: float, max_price: float) -> List[Dict]:
        """按价格范围获取菜品"""
        return [d for d in self.menu_items if min_price <= d['price'] <= max_price]


