import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# use updated HuggingFacePipeline from langchain-huggingface
# see deprecation warning in notebook
from langchain_community.llms import HuggingFacePipeline
from typing import List, Dict, Optional

import os
import requests
import json
from typing import Optional, List, Dict
import re

from log.log_utils import log_utils

# 需要把生成回复和分析意图的函数分开，避免上下文混用导致的意图分析不准确问题。

class SimpleLLM:
    """带上下文的 LLM 对话类（修正版）"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_history: int = 20, use_four_bit: bool = False, local_model: bool = True, api_key: Optional[str] = None):
        self.locl_model = local_model
        self.messages = []
        self.max_history = max_history
        self.api_key = api_key

        if self.locl_model:
            self.init_local_llm(model_name, max_history, use_four_bit)
            log_utils.d("✅ 模型加载完成")
        else:
            log_utils.d("⚠️ 使用在线API，无需本地部署")
        

    def init_local_llm(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_history: int = 10, use_four_bit: bool = False):
        log_utils.d(f"🔄 加载模型: {model_name}")
        
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log_utils.d(f"  使用设备: {self.device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.quantization_config = None
        
        # 如果试用4bit
        if use_four_bit:
            log_utils.d("⚠️ 启用4-bit量化")
            # 配置4-bit量化
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        )
        
        # 创建 pipeline（不返回完整文本），让生成参数在调用时传入以避免
        # generation_config 与显式参数混用的警告。
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False  # 重要：不返回输入文本
        )
        
        # 记录默认生成设置，后续调用 chat() 时会传给 llm.invoke
        self.generation_kwargs = {
            "max_new_tokens": 256,
            "temperature": 0.5,
            "do_sample": True,
        }

        self.llm = HuggingFacePipeline(pipeline=pipe)

    
    def add_user_message(self, content: str):
        """添加用户消息（只存原始内容）"""
        self.messages.append({"role": "user", "content": content})
        self._trim_history()
    
    def add_assistant_message(self, content: str):
        """添加助手消息（只存原始内容）"""
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()
    
    def _trim_history(self):
        """修剪历史记录"""
        if len(self.messages) > self.max_history * 2:
            self.messages = self.messages[-(self.max_history * 2):]
    
    def clear_history(self):
        """清空历史"""
        self.messages = []
    
    def set_system_prompt(self, system_prompt: str):
        """设置系统提示词（只存一次）"""
        # 移除旧的系统提示
        self.messages = [m for m in self.messages if m["role"] != "system"]
        # 添加新的系统提示到开头
        self.messages.insert(0, {"role": "system", "content": system_prompt})

    def api_llm_invoke(self, user_input: str, messages: list) -> str:    
        response = requests.post(
            'https://api.inceptionlabs.ai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            },
            json={
                'model': 'mercury-2',  
                'messages': messages,
                'max_tokens': 2000
            }
        )
        data = response.json()
        log_utils.d(f"API返回的原始数据:\n{data}\n")
        return data


    def api_llm_chat(self, user_input: str) -> str:    
        messages = self.messages.copy()
        messages.append({"role": "user", "content": user_input})
        
        data = self.api_llm_invoke(user_input, messages)

        res =  data['choices'][0]['message']['content']

        res = self._clean_response(res)

        return res
    
    def chat(self, user_input: str) -> str:
        """根据是否使用本地模型选择对话方式"""
        if self.locl_model:
            res = self.local_llm_chat(user_input)
        else:
            res = self.api_llm_chat(user_input)


        # 保存到历史
        self.add_user_message(user_input)
        self.add_assistant_message(res)

        return res
    
    def local_llm_chat(self, user_input: str) -> str:
        """
        带上下文的对话（修正版）
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI 回复（纯文本，不含模板）
        """
        # 1. 准备消息列表（只包含原始内容）
        messages = self.messages.copy()
        messages.append({"role": "user", "content": user_input})
        
        # 2. 应用聊天模板（只应用一次）
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. 生成回复，传入 generation_kwargs
        response = self.llm.invoke(prompt, **self.generation_kwargs)
        
        # 4. 清理回复（去除可能的特殊标记）
        response = self._clean_response(response)
        
        return response
    
    def chat_without_context(self, user_input: str) -> str:
        """不带上下文的对话（直接使用用户输入作为提示）"""
        messages = [{"role": "user", "content": user_input}]

        if self.locl_model:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            response = self.llm.invoke(prompt, **self.generation_kwargs)
            
            response = self._clean_response(response)
        else:
            response = self.api_llm_invoke(user_input, messages)
            response =  response['choices'][0]['message']['content']
            response = self._clean_response(response)

        return response
    
    def _clean_response(self, response: str) -> str:
        """清理回复中的特殊标记"""
        markers = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]
        for marker in markers:
            response = response.replace(marker, "")
        
        # 去除多余的空行
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        response = '\n'.join(lines)
        
        return response
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.messages.copy()
    
    # llm来分析意图
    def analyze_intent(self, user_input: str, context: Optional[Dict] = None) -> str:

        """
        意图识别 - 适配您的实际数据格式
        """
        # 构建提示词
        prompt = """你是一个餐厅外卖点餐助手，需要准确识别用户的意图。

    【意图列表及示例】
    - greeting: 打招呼、问候 (你好、在吗、嗨、您好)
    - order: 点菜、添加菜品 (我要一份、来个、点餐、来一份)
    - search: 查询菜品、问推荐 (有什么好吃的、推荐一下、有辣的吗)
    - modify_order: 修改订单 (换一个、不要洋葱、改成微辣、修改)
    - query_order: 查询订单状态 (我的订单到哪了、查一下订单、多久送到)
    - confirm_price: 确认价格 (多少钱、总共多少、价格是)
    - confirm_address: 确认地址 (送到哪里、地址对吗、收货地址)
    - place_order: 下单确认 (下单、就这些吧、确认订单)
    - farewell: 告别、感谢 (再见、拜拜、谢谢、88)
    - help: 寻求帮助 (怎么用、帮助、不会操作)
    - complaint: 投诉、不满 (太慢了、送错了、投诉、差评)

    【判断规则】
    1. 只返回意图名称(英文)，不要任何其他文字
    2. 如果完全不明确，返回 unknown
    """
        
        # 处理上下文 - 使用简单的字符串解析
        if context:
            prompt += "\n【对话历史】\n"
            try:
                # 简单解析：查找最近的对话
                # 匹配 user 和 assistant 的对话
                user_pattern = r"'role': 'user', 'content': '([^']+)'"
                assistant_pattern = r"'role': 'assistant', 'content': '([^']+)'"
                
                user_matches = re.findall(user_pattern, context)
                assistant_matches = re.findall(assistant_pattern, context)
                
                # 合并并排序（简化处理，只取最后几条）
                all_msgs = []
                for content in user_matches[-2:]:
                    all_msgs.append(f"用户: {content}")
                for content in assistant_matches[-2:]:
                    all_msgs.append(f"助手: {content}")
                
                if all_msgs:
                    prompt += '\n'.join(all_msgs[-4:]) + '\n'  # 最多4条
                else:
                    # 如果解析失败，显示原始文本的一部分
                    context_preview = context[:150] + "..." if len(context) > 150 else context
                    prompt += f"{context_preview}\n"
                    
            except Exception as e:
                log_utils.d(f"处理上下文时出错: {e}")
                context_preview = context[:150] + "..." if len(context) > 150 else context
                prompt += f"{context_preview}\n"
        
        # 添加当前输入
        prompt += f"\n【当前用户输入】\n{user_input}\n\n"
        
        log_utils.d(f"LLM分析意图的提示词:\n{prompt}\n")
        
        # 调用LLM
        res = self.chat_without_context(prompt)
        
        # 清理结果
        intent = res.strip().lower()
        # 只保留字母和短横线
        intent = re.sub(r'[^a-z-]', '', intent)
        
        # 有效意图列表
        valid_intents = ['greeting', 'order', 'search', 'modify_order', 'query_order',
                        'confirm_price', 'confirm_address', 'place_order', 'farewell',
                        'help', 'complaint', 'unknown']
        
        # 模糊匹配
        for valid in valid_intents:
            if valid in intent:
                return valid
        
        return 'unknown'
