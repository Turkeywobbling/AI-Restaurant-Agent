from llm.llm_connector import LLMConnector
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

# llm-api

class OnlineLLMConnector(LLMConnector):
    def set_api_key(self, api_key: Optional[str] = None):
        self.api_key = api_key
        log_utils.d("✅ 在线API模式，准备就绪")

    def chat(self, user_input: str) -> str:    
        if self.api_key is None:
            raise ValueError("API Key未设置，请调用set_api_key方法设置API Key")

        response = requests.post(
            'https://api.inceptionlabs.ai/v1/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            },
            json={
                'model': 'mercury-2',  
                'messages': self.messages + [{"role": "user", "content": user_input}],
                'max_tokens': 500
            }
        )
        data = response.json()
        log_utils.d(f"API返回的原始数据:\n{data}\n")

        # 保存到历史
        self.add_user_message(user_input)
        self.add_assistant_message(data['choices'][0]['message']['content'])
        return data['choices'][0]['message']['content']    