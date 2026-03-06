"""
基于向量相似度的意图识别器
使用 sentence-transformers 计算用户输入与意图示例的相似度
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple
import json
import os
from . import intentions_enum

class IntentRecognizer:
    """
    基于向量相似度的意图识别器
    
    工作原理：
    1. 每个意图有多个示例问法
    2. 将用户输入与所有示例向量化
    3. 计算相似度，取最高分
    4. 如果低于阈值，触发澄清对话
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', threshold: float = 0.8):
        """
        初始化意图识别器
        
        Args:
            model_name: 使用的embedding模型
            threshold: 置信度阈值，低于此值需要澄清
        """
        print("🔄 初始化向量意图识别器...")
        
        # 加载embedding模型
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        
        # 定义意图及其示例问法
        self.intent_examples = {
            intentions_enum.Intentions.GREETING: [
                "你好",
                "您好",
                "在吗",
                "hi",
                "hello",
                "喂",
                "你好啊",
                "晚上好",
                "上午好",
                "下午好"
            ],
            intentions_enum.Intentions.ORDER: [
                "我想点外卖",
                "点外卖",
                "我要点餐",
                "来份麻婆豆腐",
                "点一个宫保鸡丁",
                "我想要水煮肉片",
                "来一份红烧肉",
                "点菜",
                "我要吃鱼香肉丝",
                "来个西红柿炒鸡蛋",
                "给我来份酸菜鱼",
                "点一个蒜蓉西兰花"
            ],
            intentions_enum.Intentions.SEARCH: [
                "有什么好吃的",
                "推荐几个菜",
                "辣的菜有哪些",
                "便宜的菜",
                "看看菜单",
                "有什么特色菜",
                "推荐一下",
                "招牌菜是什么",
                "今天有什么推荐",
                "想吃点辣的",
                "你们店在哪",
                "有几个人",
                "你们店做什么的",
                "你们店菜品种类"
            ],
            intentions_enum.Intentions.QUERY_ORDER: [
                "看看我点了什么",
                "我的订单",
                "点了哪些菜",
                "当前订单",
                "看看点了啥",
                "订单详情",
                "我都点了什么",
                "帮我看看订单",
                "订单列表",
                "点了几个菜"
            ],
            intentions_enum.Intentions.MODIFY_ORDER: [
                "不要麻婆豆腐",
                "去掉水煮肉片",
                "把宫保鸡丁换成鱼香肉丝",
                "取消这个菜",
                "删除酸菜鱼",
                "改一下订单",
                "换成别的",
                "不要了",
                "退掉这个菜",
                "修改订单"
            ],
            intentions_enum.Intentions.CONFIRM_PRICE: [
                "多少钱",
                "一共多少钱",
                "总价多少",
                "价格是多少",
                "需要付多少",
                "多少钱啊",
                "总共多少钱",
                "算一下价格",
                "多少钱一份",
                "这个多少钱"
            ],
            intentions_enum.Intentions.CONFIRM_ADDRESS: [
                "送到哪里",
                "配送地址",
                "我的地址是",
                "送到",
                "地址",
                "在什么地方",
                "位置",
                "送餐地址",
                "家在",
                "公司地址"
            ],
            intentions_enum.Intentions.CONFIRM_PHONE: [
                "电话",
                "手机号",
                "联系电话",
                "我的电话是",
                "手机号码",
                "联系方式",
                "电话多少",
                "留个电话",
                "号码",
                "怎么联系"
            ],
            intentions_enum.Intentions.PLACE_ORDER: [
                "下单",
                "确认订单",
                "就这些",
                "点完了",
                "结账",
                "买单",
                "提交订单",
                "确认下单",
                "可以了",
                "就点这些"
            ],
            intentions_enum.Intentions.FAREWELL: [
                "再见",
                "拜拜",
                "谢谢",
                "感谢",
                "下次再点",
                "88",
                "bye",
                "goodbye",
                "拜",
                "多谢"
            ],
            intentions_enum.Intentions.HELP: [
                "帮助",
                "怎么点餐",
                "如何使用",
                "help",
                "功能",
                "能做什么",
                "你会什么",
                "怎么用",
                "介绍一下",
                "说明"
            ],
            intentions_enum.Intentions.COMPLAINT: [
                "太慢了",
                "不好吃",
                "投诉",
                "不满意",
                "差评",
                "怎么这么慢",
                "味道不对",
                "送错了",
                "少送了",
                "有问题"
            ]
        }
        
        # 预计算所有示例的向量
        self._precompute_example_vectors()
        
        # 意图别名映射（有些意图可能表达相似）
        self.intent_aliases = {
            "order": ["order", "add_to_order", "place"],
            "search": ["search", "query", "find", "recommend"],
            "query_order": ["query_order", "check_order", "view_order"],
            "modify_order": ["modify", "change", "remove", "delete"],
            "confirm_price": ["price", "cost", "total"],
            "confirm_address": ["address", "location", "delivery"],
            "confirm_phone": ["phone", "contact", "mobile"],
            "place_order": ["checkout", "submit", "confirm"]
        }
        
        print(f"✅ 意图识别器初始化完成，共 {len(self.intent_examples)} 个意图，{self._total_examples()} 个示例")
    
    def _total_examples(self) -> int:
        """计算总示例数"""
        return sum(len(examples) for examples in self.intent_examples.values())
    
    def _precompute_example_vectors(self):
        """预计算所有示例的向量"""
        self.intent_vectors = {}
        
        for intent, examples in self.intent_examples.items():
            # 为每个意图的示例计算向量
            vectors = self.model.encode(examples)
            self.intent_vectors[intent] = {
                'examples': examples,
                'vectors': vectors,
                'mean_vector': np.mean(vectors, axis=0)  # 平均向量，用于快速匹配
            }
        
        print(f"  预计算 {len(self.intent_vectors)} 个意图的向量")
    
    def recognize(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        识别用户意图
        
        Args:
            text: 用户输入文本
            context: 对话上下文，可用于调整意图判断
            
        Returns:
            包含意图和置信度的字典
        """
        # 生成用户输入的向量
        user_vector = self.model.encode(text)
        
        # 计算与每个意图的相似度
        similarities = {}
        detailed_matches = {}
        
        # SentenceTransformer 默认输出已归一化向量（范数≈1），因此
        # 余弦相似度可以直接用点积表示。无需额外归一化步骤。
        for intent, data in self.intent_vectors.items():
            # 直接计算用户向量与每个示例的点积
            dot_products = np.dot(data['vectors'], user_vector)
            example_sims = dot_products  # 已经是余弦相似度
            
            # 取最高相似度
            max_sim = np.max(example_sims)
            similarities[intent] = float(max_sim)
            
            # 记录最匹配的示例
            best_idx = np.argmax(example_sims)
            detailed_matches[intent] = {
                'best_example': data['examples'][best_idx],
                'similarity': float(max_sim)
            }
                
        # 找出最佳意图
        best_intent = max(similarities, key=similarities.get)
        best_score = similarities[best_intent]
        
        # 考虑上下文调整（如果有）
        if context:
            best_intent, best_score = self._adjust_with_context(
                best_intent, best_score, similarities, context
            )
        
        # 判断是否需要澄清
        needs_clarification = best_score < self.threshold
        
        # 找出备选意图（得分第二、第三高的）
        sorted_intents = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        alternatives = [
            {"intent": intent, "score": score}
            for intent, score in sorted_intents[1:4]  # 取2-4名
            if score > self.threshold * 0.8  # 只考虑不太低的
        ]
        
        result = {
            "intent": best_intent,
            "confidence": best_score,
            "needs_clarification": needs_clarification,
            "alternatives": alternatives,
            "detailed_matches": detailed_matches[best_intent],
            "all_scores": similarities
        }
        
        return result
    
    def _adjust_with_context(self, best_intent: str, best_score: float, 
                            all_scores: Dict, context: Dict) -> Tuple[str, float]:
        """
        根据上下文调整意图判断
        
        Args:
            best_intent: 当前最佳意图
            best_score: 当前最佳得分
            all_scores: 所有意图得分
            context: 上下文信息，包含历史意图、当前阶段等
        """
        adjusted_intent = best_intent
        adjusted_score = best_score
        
        # 获取上一轮意图
        last_intent = context.get('last_intent')
        current_stage = context.get('stage')
        
        # 如果当前阶段明确，提高相关意图的权重
        if current_stage:
            stage_intent_map = {
                'confirming_price': intentions_enum.Intentions.CONFIRM_PRICE,
                'confirming_address': intentions_enum.Intentions.CONFIRM_ADDRESS,
                'placing_order': intentions_enum.Intentions.PLACE_ORDER
            }
            
            if current_stage in stage_intent_map:
                expected_intent = stage_intent_map[current_stage]
                if expected_intent in all_scores:
                    # 提高预期意图的得分
                    adjusted_score = all_scores[expected_intent] * 1.2
                    if adjusted_score > best_score * 1.1:
                        adjusted_intent = expected_intent
        
        # 如果上一轮是确认类意图，这一轮可能是确认回答
        if last_intent in [intentions_enum.Intentions.CONFIRM_PRICE, intentions_enum.Intentions.CONFIRM_ADDRESS, intentions_enum.Intentions.PLACE_ORDER]:
            confirm_intent = 'confirm'  # 可能需要一个确认意图
            # 检查是否是肯定回答
            confirm_words = ['是', '对', '好', '可以', '行', '嗯', 'ok']
            if context.get('user_input') and any(word in context['user_input'] for word in confirm_words):
                # 继续当前阶段
                adjusted_intent = last_intent
                adjusted_score = max(adjusted_score, 0.8)
        
        return adjusted_intent, adjusted_score
    
    def generate_clarification(self, text: str, recognition_result: Dict) -> str:
        """
        生成澄清问题
        
        Args:
            text: 用户输入
            recognition_result: 意图识别结果
            
        Returns:
            澄清问题
        """
        alternatives = recognition_result['alternatives']
        
        if not alternatives:
            return "抱歉，我没太明白您的意思。您是想点餐、查询菜单，还是其他什么？"
        
        # 根据备选意图生成澄清问题
        intent_questions = {
            intentions_enum.Intentions.ORDER.value: "您是想点餐吗？",
            intentions_enum.Intentions.SEARCH.value: "您是想查询菜单吗？",
            intentions_enum.Intentions.QUERY_ORDER.value: "您是想查看订单吗？",
            intentions_enum.Intentions.MODIFY_ORDER.value: "您是想修改订单吗？",
            intentions_enum.Intentions.CONFIRM_PRICE.value: "您是想询问价格吗？",
            intentions_enum.Intentions.CONFIRM_ADDRESS.value: "您是想确认配送地址吗？",
            intentions_enum.Intentions.PLACE_ORDER.value: "您是想下单吗？"
        }
        
        # 取得分最高的备选意图
        top_alternative = alternatives[0]['intent']
        
        if top_alternative.value in intent_questions:
            return intent_questions[top_alternative]
        else:
            return "请问您需要什么帮助？点餐、查询还是其他服务？"
    
    def add_example(self, intent: str, example: str):
        """
        动态添加新的示例（用于在线学习）
        
        Args:
            intent: 意图名称
            example: 新的示例问法
        """
        if intent not in self.intent_examples:
            self.intent_examples[intent] = []
        
        self.intent_examples[intent].append(example)
        
        # 重新计算该意图的向量
        examples = self.intent_examples[intent]
        vectors = self.model.encode(examples)
        self.intent_vectors[intent] = {
            'examples': examples,
            'vectors': vectors,
            'mean_vector': np.mean(vectors, axis=0)
        }
        
        print(f"✅ 为意图 '{intent}' 添加新示例: '{example}'")
    
    def get_statistics(self) -> Dict:
        """获取识别器统计信息"""
        stats = {
            'total_intents': len(self.intent_examples),
            'total_examples': self._total_examples(),
            'intent_details': {}
        }
        
        for intent, examples in self.intent_examples.items():
            stats['intent_details'][intent] = {
                'example_count': len(examples),
                'examples': examples[:3]  # 只显示前3个示例
            }
        
        return stats

