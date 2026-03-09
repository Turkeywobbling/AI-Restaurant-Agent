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
from llm.llm_connector import LLMConnector
from log.log_utils import log_utils

class LocalLLMDeployer(LLMConnector):
    def init_local_llm(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", use_four_bit: bool = False):
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

    def chat(self, user_input: str) -> str:
        """
        带上下文的对话（修正版）
        
        Args:
            user_input: 用户输入
            
        Returns:
            AI 回复（纯文本，不含模板）
        """

        if self.llm is None:
            raise ValueError("模型未初始化，请先调用 init_local_llm 方法加载模型")

        # 1. 准备消息列表（只包含原始内容）
        messages = self.messages.copy()
        
        # 2. 应用聊天模板（只应用一次）
        prompt = self.tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": user_input}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. 生成回复，传入 generation_kwargs
        response = self.llm.invoke(prompt, **self.generation_kwargs)
        
        # 4. 清理回复（去除可能的特殊标记）
        response = self._clean_response(response)

        # 保存到历史
        self.add_user_message(user_input)
        self.add_assistant_message(response)
        
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
    