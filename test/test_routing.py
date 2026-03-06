"""诊断脚本：测试路由和节点执行"""
import sys
from order_agent import RestaurantAgent, AgentState
from menu import Menu
from IntentRecognizer import IntentRecognizer
from NLPEntityExtractor import NLPEntityExtractor

# 模拟依赖（根据你的实际实现调整）
class MockLLM:
    def chat(self, prompt):
        return "greeting"

class MockVectorStore:
    def search(self, query, k=5):
        return []

# 创建 agent
menu = Menu()
llm = MockLLM()
vector_store = MockVectorStore()
intent_recognizer = IntentRecognizer()
entity_extractor = NLPEntityExtractor()

agent = RestaurantAgent(menu, llm, vector_store, intent_recognizer, entity_extractor)

# 打印图的结构
print("=" * 50)
print("🔍 LangGraph 结构诊断")
print("=" * 50)

print("\n1️⃣ 条件边映射:")
print("   intent_recognition → (conditional edges)")
conditional_config = {
    "needs_clarification": "handle_clarification",
    "search": "search_menu",
    "order": "process_order",
    "modify_order": "process_order",
    "query_order": "process_order",
    "confirm_price": "confirm_price",
    "confirm_address": "confirm_address",
    "place_order": "place_order",
    "greeting": "generate_response",
    "farewell": "generate_response",
    "help": "generate_response",
    "default": "generate_response"
}
for key, value in conditional_config.items():
    print(f"   {key:20} → {value}")

print("\n2️⃣ 路由函数测试:")
test_states = [
    {"requires_clarification": True, "stage": "ordering", "intent": "greeting"},
    {"requires_clarification": False, "stage": "ordering", "intent": "greeting"},
    {"requires_clarification": False, "stage": "confirming_address", "intent": "order"},
]

for test_state in test_states:
    result = agent.route_by_intent(test_state)
    print(f"   State {test_state} → {result}")
    if result not in conditional_config:
        print(f"   ⚠️  警告: '{result}' 不在 conditional_config 中!")

print("\n3️⃣ 执行完整流程:")
print("   输入: '你好'")
try:
    response = agent.process("你好")
    print(f"   输出: {response}")
    print("   ✅ 执行成功")
except Exception as e:
    print(f"   ❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
