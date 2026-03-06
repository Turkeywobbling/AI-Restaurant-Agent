"""快速测试脚本"""
import sys
sys.path.insert(0, r'c:\Users\PC\anaconda_projects\ai_food_order')

print("=" * 60)
print("🧪 LangGraph 图编译和执行测试")
print("=" * 60)

try:
    print("\n1️⃣ 导入模块...")
    from order_agent import RestaurantAgent, AgentState
    print("   ✅ 导入成功")
    
    print("\n2️⃣ 检查 AgentState...")
    print(f"   - AgentState 字段: {list(AgentState.__annotations__.keys())[:5]}...")
    print(f"   - messages 类型: {AgentState.__annotations__.get('messages')}")
    print("   ✅ AgentState 定义正确")
    
    print("\n3️⃣ 模拟依赖项...")
    class MockLLM:
        def chat(self, prompt):
            if "意图" in prompt or "识别" in prompt:
                return "greeting"
            return ""
    
    class MockMenu:
        def exists(self, name):
            return True
        def get_price(self, name):
            return 10.0
        def get_dish(self, name):
            return {"name": name, "price": 10.0}
    
    class MockVectorStore:
        def search(self, query, k=5):
            return []
    
    class MockIntentRecognizer:
        def recognize(self, text, context):
            return {
                "intent": "greeting",
                "confidence": 1.0,
                "needs_clarification": False
            }
        def generate_clarification(self, text, result):
            return "请问您想要什么？"
    
    class MockEntityExtractor:
        def extract(self, text):
            return {}
    
    print("   ✅ 模拟依赖项创建成功")
    
    print("\n4️⃣ 创建 RestaurantAgent...")
    agent = RestaurantAgent(
        menu=MockMenu(),
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        intent_recognizer=MockIntentRecognizer(),
        entity_extractor=MockEntityExtractor()
    )
    print("   ✅ Agent 创建成功")
    
    print("\n5️⃣ 测试图编译...")
    print(f"   - 图对象: {agent.graph}")
    print(f"   - 图类型: {type(agent.graph)}")
    print("   ✅ 图编译成功")
    
    print("\n6️⃣ 测试单个流程...")
    response = agent.process("你好")
    print(f"   ✅ 响应: {response[:50]}...")
    
    print("\n" + "=" * 60)
    print("✨ 所有测试通过！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
