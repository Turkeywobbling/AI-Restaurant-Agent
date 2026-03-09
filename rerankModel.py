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

from log import log_utils

class HybridRerankStore:
    """混合重排序 - 规则 + 模型"""
    
    def __init__(self, vectorDB):
        self.vector_store = vectorDB
        
        # 加载轻量级rerank模型
        try:
            from sentence_transformers import CrossEncoder
            # 英文菜单 self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.reranker = CrossEncoder(
                'BAAI/bge-reranker-base',  # 中文基础版
                # 'BAAI/bge-reranker-large',  # 中文大模型（更好但更慢）
                max_length=512,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            self.use_model = True
            log_utils.d("rerank模型加载完成")
        except Exception as e:
            log_utils.d(f"❌ 加载rerank模型失败: {e}")
            self.use_model = False
            log_utils.d("使用规则重排序")
    
    def _rule_score(self, dish: dict, query: str) -> float:
        """规则分数"""
        score = 0
        query = query.lower()
        
        # 辣度匹配
        if "辣" in query:
            spicy_map = {"重辣": 1.0, "中辣": 0.8, "微辣": 0.5, "不辣": 0.0}
            score += spicy_map.get(dish['spicy_level'], 0) * 0.3
        
        # 价格匹配
        if "便宜" in query and dish['price'] < 30:
            score += 0.2
        if "贵" in query and dish['price'] > 50:
            score += 0.2
        
        # 菜品名匹配
        if dish['name'] in query:
            score += 0.5
        
        return score
    
    def search(self, query: str, k: int = 5):
        """搜索 + 混合Rerank"""
        
        # 1. 获取候选
        candidates = self.vector_store.search(query, k=20)
        
        # 2. 计算分数
        for dish in candidates:
            # 向量相似度分数
            vec_score = dish['similarity']
            
            # 规则分数
            rule_score = self._rule_score(dish, query)
            
            # 模型分数（如果有）
            if self.use_model:
                model_score = self.reranker.predict([[query, dish['name']]])[0]
                # 混合分数
                dish['final_score'] = vec_score * 0.3 + rule_score * 0.3 + model_score * 0.4
            else:
                # 只有向量+规则
                dish['final_score'] = vec_score * 0.6 + rule_score * 0.4
        
        # 3. 排序
        results = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        return results[:k]

