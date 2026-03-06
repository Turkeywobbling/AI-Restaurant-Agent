"""测试图的完整流程"""
import sys
import os
sys.path.insert(0, r'c:\Users\PC\anaconda_projects\ai_food_order')

# 禁用其他模块的日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 70)
print("🧪 LangGraph 完整流程测试")
print("=" * 70)

try:
    print("\n📦 导入模块...")
    from order_agent import RestaurantAgent, AgentState
    
    print("\n🔧 创建mock对象...")
    
    class MockLLM:
        def chat(self, prompt):
            return "greeting"
    
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
    
    print("✅ Mock对象创建完成\n")
    
    print("🤖 创建 RestaurantAgent...")
    agent = RestaurantAgent(
        menu=MockMenu(),
        llm=MockLLM(),
        vector_store=MockVectorStore(),
        intent_recognizer=MockIntentRecognizer(),
        entity_extractor=MockEntityExtractor()
    )
    print("✅ Agent创建完成\n")
    
    print("📊 图的结构:")
    print(f"   - 编译的图: {agent.graph}")
    print()
    
    print("🚀 执行完整的用户输入流程: '你好'")
    print("-" * 70)
    
    response = agent.process("你好")
    
    print("-" * 70)
    print(f"\n✨ 最终响应:\n{response}\n")
    
    print("=" * 70)
    print("✅ 测试成功！图能够正常执行完整流程")
    print("=" * 70)
    
except Exception as e:
    print(f"\n❌ 错误: {type(e).__name__}: {e}\n")
    import traceback
    traceback.print_exc()
    print("\n" + "=" * 70)
