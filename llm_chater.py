import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# use updated HuggingFacePipeline from langchain-huggingface
# see deprecation warning in notebook
from langchain_community.llms import HuggingFacePipeline
from typing import List, Dict, Optional

class SimpleLLM:
    """带上下文的 LLM 对话类（修正版）"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", max_history: int = 10, use_four_bit: bool = False):
        print(f"🔄 加载模型: {model_name}")
        
        self.max_history = max_history
        self.messages = []  # 只存储原始内容，不存储模板格式
        
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  使用设备: {self.device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.quantization_config = None
        
        # 如果试用4bit
        if use_four_bit:
            print("⚠️ 启用4-bit量化")
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
        print("✅ 模型加载完成")
    
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
    
    def chat(self, user_input: str) -> str:
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
        
        # 5. 保存到历史
        self.add_user_message(user_input)
        self.add_assistant_message(response)
        
        return response
    
    def chat_without_context(self, user_input: str) -> str:
        """不带上下文的对话（直接使用用户输入作为提示）"""
        messages = [{"role": "user", "content": user_input}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        response = self.llm.invoke(prompt, **self.generation_kwargs)
        
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
        prompt = "你是餐厅的ai线上点外卖助手，请帮忙识别用户所说话的意图， 可能的意图有：" 
        prompt += "greeting（打招呼）, order（点菜）, search（问菜）, modify_order（修改订单）, query_order（查询订单）, confirm_price（确认价格）, confirm_address（确认地址）, place_order（下单）, farewell（告别）, help（帮助）， complaint（投诉）。"
        

        prompt += "请直接返回意图名称(英文)，不要其他任何文本。如果你也不确定，请返回unknown。 " 
        
        if context:
            prompt += "你与用户的对话信息: " + str(context) + "。"
        else:
            prompt += "用户输入是: " + user_input

        print(f"LLM分析意图的提示词:\n{prompt}\n")

        res = self.chat_without_context(prompt)
        return res
