from abc import ABC, abstractmethod
from typing import Dict, Optional

from log.log_utils import log_utils

# 仅用于链接llm

class LLMConnector(ABC):

    def __init__(self):
        self.messages = []
        self.system_prompt_set = False
        
        
    def add_user_message(self, content: str):
        """添加用户消息（只存原始内容）"""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str):
        """添加助手消息（只存原始内容）"""
        self.messages.append({"role": "assistant", "content": content})

    def clear_history(self):
        """清空历史"""
        self.messages = []

    def set_system_prompt(self, system_prompt: str):
        """设置系统提示词（只存一次）"""
        # 移除旧的系统提示
        if self.system_prompt_set:
            log_utils.d("⚠️ 已存在系统提示")
        # 添加新的系统提示到开头
        self.messages.insert(0, {"role": "system", "content": system_prompt})

        self.system_prompt_set = True


    @abstractmethod
    def chat(self, user_input: str) -> str:
        pass

    
    def analyze_intent(self, user_input: str) -> str:
        """
        通过读取外部文件构建意图识别提示。

        当前实现仅负责把存放在 ``llm/intention_reg_promp.txt`` 的模板读入为一个字符串，
        并将用户输入（以及可选上下文）附加到该字符串末尾后返回。

        子类或调用方可以使用返回的文本作为发送给 LLM 的完整提示；
        如果不需要额外处理，则直接返回提示内容。这样可以保证
        基类在不知晓具体 LLM 调用细节的情况下提供统一的提示文本。
        """
        import os
        if not self.system_prompt_set:
            # locate the prompt file relative to this module
            prompt_path = os.path.join(os.path.dirname(__file__), "intention_reg_promp.txt")
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    template = f.read()
            except Exception as e:
                # 如果文件读取失败，抛出异常以便上层感知
                raise RuntimeError(f"读取意图识别提示文件失败: {e}")
            
            self.set_system_prompt(template)

        # 构造最终提示
        prompt = "现在，请对以下用户输入进行意图识别，保持输出内容格式的一致性，对话历史为："

        # 处理消息历史，提取用户和助手的对话内容，简化为纯文本形式
        ctx = ""
        for msg in self.messages:
            if msg["role"] == "user":
                ctx += f"\n用户: {msg['content']}"
            elif msg["role"] == "assistant":
                ctx += f"\n助手: {msg['content']}"


        prompt += ctx
        prompt += "\n\n【当前用户输入】\n" + user_input

        res = self.chat(prompt)

        return res

        
