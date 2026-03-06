import os
import json
from typing import Dict, List, TypedDict, Optional
from datetime import datetime

from customer import Customer
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import traceback
import random

import order
import stage_enum
import intention.intentions_enum as intentions_enum

# todo 统一阶段（意图）名称！！！！
# todo 需要知道上一节点是什么，比如点了一个菜后，是结账还是继续点？

class AgentState(TypedDict, total=False):
    """Agent状态"""
    messages: List[Dict]  # 对话历史（移除 Annotated）
    user_input: str  # 当前用户输入
    intent_result: Dict  # 意图识别结果
    
    # 意图和改写
    intent: intentions_enum.Intentions  # 识别出的意图
    intent_confidence: float  # 意图置信度
    confidence: float  # 意图置信度（兼容旧命名）
    entities: Dict  # 提取的实体
    rewritten_query: str  # 改写后的查询
    
    # 搜索结果
    search_results: List  # 搜索结果
    selected_dish: Optional[Dict]  # 当前选中的菜品
    
    # 订单相关
    current_order: Optional[order.order_manager]  # 当前订单
    order_total: float  # 订单总价
    order_confirmed: bool  # 订单是否已确认
    
    # 客户信息
    customer_info: Dict  # 客户信息
    
    # 流程控制
    stage: stage_enum.stage  # 当前对话阶段: ordering, confirming_price, confirming_address, placing_order, completed
    requires_clarification: bool  # 是否需要澄清
    clarification_question: Optional[str]  # 澄清问题
    
    # 最终回复
    response: str

# ============ RestaurantAgent 实现 ============

class RestaurantAgent:
    """餐厅点餐Agent - 使用向量意图识别"""
    
    def __init__(self, menu, llm, vector_store, intent_recognizer, entity_extractor):
        self.menu = menu
        self.vector_store = vector_store
        self.llm = llm
        
        
        # 使用向量意图识别器
        self.intent_recognizer = intent_recognizer
        self.entity_extractor = entity_extractor
        
        # 构建图
        self.graph = self._build_graph()

        # 对话历史
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.order_manager = order.order_manager(self.session_id)

        # 意图识别升降级
        self.is_llm_analyze_intent = False
        
        print("\n✅ RestaurantAgent 初始化完成")
    
    def _build_graph(self):
        """构建LangGraph"""
        
        workflow = StateGraph(AgentState)
        
        # 包装节点函数以添加追踪
        def trace_node(node_func):
            def wrapped(state):
                print(f"\n🔷 进入节点: {node_func.__name__}")
                print(f"  当前意图: {state.get('intent')}, 阶段: {state.get('stage')}")
                try:
                    result = node_func(state)
                    print(f"  ✅ 退出节点: {node_func.__name__}")
                    if result.get('response'):
                        print(f"  响应: {result['response'][:50]}...")
                    return result
                except Exception as e:
                    print(f"  ❌ 节点异常: {e}")
                    raise
            return wrapped
        
        # 添加节点（使用包装后的函数）
        workflow.add_node("intent_recognition", trace_node(self.intent_recognition_node))
        workflow.add_node("handle_clarification", trace_node(self.handle_clarification_node))
        workflow.add_node("search_menu", trace_node(self.search_menu_node))
        workflow.add_node("process_order", trace_node(self.process_order_node))
        workflow.add_node("confirm_price", trace_node(self.confirm_price_node))
        workflow.add_node("confirm_address", trace_node(self.confirm_address_node))
        workflow.add_node("place_order", trace_node(self.place_order_node))
        # workflow.add_node("chichat", trace_node(self.generate_response_node))
        workflow.add_node("generate_response", trace_node(self.generate_response_node))
        
        # 设置入口
        workflow.set_entry_point("intent_recognition")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "intent_recognition",
            self.route_by_intent,
            {
                "handle_clarification": "handle_clarification",
                "search_menu": "search_menu",
                "process_order": "process_order",
                "confirm_price": "confirm_price",
                "confirm_address": "confirm_address",
                "place_order": "place_order",
                "generate_response": "generate_response"
            }
        )
        
        workflow.add_edge("handle_clarification", "generate_response")
        workflow.add_edge("search_menu", "generate_response")
        workflow.add_edge("process_order", "generate_response")
        workflow.add_edge("confirm_price", "generate_response")
        workflow.add_edge("confirm_address", "generate_response")
        workflow.add_edge("place_order", "generate_response")
        workflow.add_edge("generate_response", END)

        graph = workflow.compile()

        return graph
    
    # 封装一个agent与用户对话的输出，与llm的返回做区分
    def agent_response(self, response: str) -> str:
        """封装Agent回复"""
        res = f"Agent回复: {response}"
        return res
    
    # ============ 节点实现 ============

    def llm_intent_recognition(self, user_input: str) -> str:
        return self.llm.analyze_intent(user_input)
    
    def intent_recognition_node(self, state: AgentState) -> AgentState: # todo!! 增加升降级，IntentRecognizer准确度低的时候，改用llm分析意图
        """意图识别节点 - 使用向量识别器"""
        user_input = state["user_input"]
        current_stage = state.get("stage", stage_enum.stage.ORDERING)
        
        # 构建上下文
        context = {
            'last_intent': state.get('intent'),
            'stage': current_stage,
            'user_input': user_input
        }
        
        if self.is_llm_analyze_intent: # 升降级
            print("⚠️ 当前使用LLM分析意图")
            intent_result = self.llm_intent_recognition(user_input)
            print(f"LLM分析意图原始结果: {intent_result}")
            if intent_result != "unknown":
                if intent_result in intentions_enum.Intentions.__members__:
                    intent_result = intentions_enum.Intentions[intent_result]

                print(f"  ✅ LLM识别意图: {intent_result}")
                state["intent"] = intent_result
                state["requires_clarification"] = False
                return state
            else:
                print(f"  ❌ LLM也无法识别意图，保持需要澄清的状态")
                state["requires_clarification"] = True
                return state
        else:
            # 向量意图识别
            intent_result = self.intent_recognizer.recognize(user_input, context)
        
            # 保存结果
            state["intent_result"] = intent_result
            state["intent"] = intent_result["intent"]
            state["confidence"] = intent_result["confidence"]
            state["requires_clarification"] = intent_result["needs_clarification"]
            
            # 打印调试信息
            print("用户输入： " + user_input)
            print(f"\n意图识别:")
            print(f"  意图: {intent_result['intent']} (置信度: {intent_result['confidence']:.1%})")
            
            if intent_result['needs_clarification']:
                print("置信度低，使用大模型分析意图")
                llm_intent_recognition_result = self.llm_intent_recognition(user_input)

                if llm_intent_recognition_result != "unknown":
                    print(f"  ✅ LLM识别意图: {llm_intent_recognition_result}")
                    state["intent"] = llm_intent_recognition_result
                    state["requires_clarification"] = False
                
                else:
                    print(f"  ❌ LLM也无法识别意图，保持需要澄清的状态")
                    state["requires_clarification"] = True
            
            print("intent_recognition_node, 准备进入下一节点: " + self.route_by_intent(state))
            return state
    
    def handle_clarification_node(self, state: AgentState) -> AgentState:
        """处理需要澄清的情况"""
        intent_result = state["intent_result"]
        user_input = state["user_input"]
        
        # 生成澄清问题
        clarification = self.intent_recognizer.generate_clarification(
            user_input, intent_result
        )
        
        state["clarification_question"] = clarification
        state["response"] = clarification
        
        return state
    
    def search_menu_node(self, state: AgentState) -> AgentState: # 有问题todo
        """搜索菜单节点"""
        user_input = state["user_input"]
        
        # 执行搜索
        results = self.vector_store.search(user_input, k=5)
        state["search_results"] = results
        
        print(f"🔍 找到 {len(results)} 个相关菜品")
        
        return state
    
    def verify_dish(self, dish_name) -> bool:
        # 检查dish是否是菜单里的
        if dish_name in self.menu.dishes:
            return True
        else:
            return False
        

    def vectorDB_search_dish(self, dish_name, threshold=0.7) -> str:
        results = self.vector_store.search(dish_name)
        print("菜品向量搜索: ")
        print(results)
        matched_dish = results[0]
        if matched_dish['similarity'] < threshold:
            return ""
        else:
            return matched_dish['name']

    def process_order_node(self, state: AgentState) -> AgentState:
        """处理订单节点"""
        
        intent = state["intent"]
        user_input = state["user_input"]
        
        # 初始化订单状态（如果没有）
        if state.get("current_order") is None:
            state["current_order"] = order.order_manager(id=1)
        
        if intent == intentions_enum.Intentions.ORDER:
            # 先用entityExtracter找菜名
            selected_dish = self.entity_extractor.extract(user_input)
            
            # 如果entityExtracter没找到菜名，或者用户输入的菜名不在菜单里，就用LLM辅助提取菜名和数量
            if len(selected_dish) == 0:
                # LLM辅助提取
                agent_input = "用户似乎想要点餐，请帮忙找出他想要点的菜品和数量。" \
                "如果有菜品的话，请以python dict的结构返回，key是菜名，value是数量; 如果没有，则返回[]。不要回复其他的。" \
                "用户原话: " + user_input
                
                llm_res = self.llm.chat(agent_input)
                raw_order_data = {}
                try:
                    raw_order_data = eval(llm_res)
                    print(f"📦 LLM提取结果: {raw_order_data}")
                except Exception as e:
                    print(f"{e}: " + llm_res)
                
                if raw_order_data and len(raw_order_data) > 0:
                    # 处理LLM提取的结果
                    for dish_name, quantity in raw_order_data.items():
                        self._add_dish_to_order(state, dish_name, quantity)
                else:
                    # 兜底：仍然没找到菜品
                    state["needs_clarification"] = True
                    state["clarification_question"] = "抱歉，我没理解您要点什么菜。请告诉我具体的菜名，比如'麻婆豆腐'。"
                    return state
            else:
                # entityExtracter直接找到了
                dish_name = selected_dish.get('dish')
                quantity = selected_dish.get('quantity', 1)
                self._add_dish_to_order(state, dish_name, quantity)
        
        elif intent == intentions_enum.Intentions.MODIFY_ORDER:
            # 修改订单
            selected_dish = self.entity_extractor.extract(user_input)
            if selected_dish.get('dish'):
                dish_name = selected_dish['dish']
                # 从订单中移除
                if state.get("current_order"):
                    price = self.menu.get_price(dish_name)
                    state["current_order"].remove_item(dish_name, price)
                    self._update_order_total(state)
                    print(f"❌ 从订单移除: {dish_name}")
        
        elif intent == intentions_enum.Intentions.QUERY_ORDER:
            # 查询订单 - 不需要修改，只是返回当前状态
            pass
        
        # 更新订单总价和状态
        self._update_order_total(state)
        
        # 根据订单状态更新阶段
        if state.get("current_order") and len(state["current_order"].order_data["items"]) > 0 and state.get("order_stage") == "ordering":
            state["order_stage"] = "confirm_address"
        
        # 打印当前订单状态
        self._print_order_status(state)
        
        return state
    
    def _add_dish_to_order(self, state: AgentState, dish_name: str, quantity: int = 1):
        """添加菜品到订单的辅助方法"""
        
        # 1. 先验证菜品名是否在菜单中
        if self.menu.exists(dish_name):
            # 直接存在
            price = self.menu.get_price(dish_name)
            dish_info = self.menu.get_dish(dish_name)
        else:
            # 2. 不在菜单中，用向量数据库搜索
            searched_dish = self._search_dish_in_db(dish_name)
            if searched_dish:
                dish_name = searched_dish['name']
                price = searched_dish['price']
                dish_info = searched_dish
            else:
                # 3. 完全找不到
                print(f"⚠️ 未找到菜品: {dish_name}")
                state["needs_clarification"] = True
                state["clarification_question"] = f"抱歉，我们没有找到'{dish_name}'，您看看其他菜品？"
                return
        
        # 新增订单项
        current_items = state["current_order"]
        dish_price = self.menu.get_price(dish_name)
        current_items.add_item(dish_name, dish_price, quantity)

        
        state["current_order"] = current_items
        print(f"✅ 添加新菜品: {dish_name}")

    def _search_dish_in_db(self, dish_name: str) -> dict:
        """在向量数据库中搜索菜品"""
        try:
            results = self.vector_store.search(dish_name, k=1)
            if results and len(results) > 0:
                return {
                    'name': results[0]['name'],
                    'price': results[0]['price'],
                    'description': results[0].get('description', '')
                }
        except Exception as e:
            print(f"搜索菜品失败: {e}")
        
        return None 

    def _update_order_total(self, state: AgentState):
        """更新订单总价"""
        if state.get("current_order"):
            state["order_total"] = state["current_order"].order_data["total_price"]

    def _print_order_status(self, state: AgentState):
        """打印订单状态"""
        if state.get("current_order") and len(state["current_order"].order_data["items"]) > 0:
            print("\n📋 当前订单:")
            for item_name, quantity in state["current_order"].order_data["items"].items():
                price = self.menu.get_price(item_name)
                subtotal = price * quantity
                print(f"  {item_name} x{quantity} = {subtotal}元")
            print(f"💰 总计: {state.get('order_total', 0)}元")   

    def llm_classify_intent(self, user_input: str) -> intentions_enum.Intentions:
        """使用LLM辅助意图分类"""
        prompt = "你是餐厅的AI线上外卖客服，负责理解用户的意图，需要你来判断用户是确认订单没问题还是有问题,用户输入是: " + user_input + "。如果用户的输入中包含了确认订单的意图，请回复'confirm_price'，如果用户的输入中包含了继续点餐的意图，请回复'modify_order'，如果无法判断，请回复'unknown'。不要回复其他的。"
        response = self.llm.chat(prompt)
        if "confirm" in response.lower() or "确认" in response:
            return intentions_enum.Intentions.CONFIRM_PRICE
        elif "继续" in response.lower() or "点餐" in response:
            return intentions_enum.Intentions.MODIFY_ORDER
        else:
            return intentions_enum.Intentions.UNKNOWN
    
    def confirm_price_node(self, state: AgentState) -> AgentState: # 是否需要?? todo
        """确认价格节点"""
        user_input = state["user_input"]

        # 改用llm来判断用户是否确认价格，避免用户说“确认”但实际是想继续点餐的情况
        confirmed_intent = self.llm_classify_intent(user_input)
        if confirmed_intent != intentions_enum.Intentions.UNKNOWN:
            if confirmed_intent == intentions_enum.Intentions.CONFIRM_PRICE:
                state["order_confirmed"] = True
                state["stage"] = stage_enum.stage.CONFIRMING_ADDRESS
                state["requires_clarification"] = False
            elif confirmed_intent == intentions_enum.Intentions.MODIFY_ORDER:
                state["stage"] = stage_enum.stage.ORDERING
                state["requires_clarification"] = False
            else:
                state["requires_clarification"] = True
                state["clarification_question"] = "请问您是想确认订单还是继续点餐？"        
        
        return state
    
    def confirm_address_node(self, state: AgentState) -> AgentState:
        """确认地址节点"""

        # 先用llm对话的方式来获得用户地址
        # 后续可以将用户信息进行存储，常配送地直接读取
        res = self.agent_response("收到，请问您的配送地址是？")
        print(res) # 暂时交互用

        location = input("请输入地址: ").strip() # 这里直接用input模拟用户输入地址，后续可以改成对话交互
        llm_extracted_location = self.llm.chat("请从用户输入中提取配送地址，用户输入是: " + location + "。如果能提取到地址，请直接返回地址文本，如果无法提取，请返回空字符串。")
        if llm_extracted_location:
            current_order = state["current_order"]
            current_order.set_delivery_location(llm_extracted_location)
        else:
            print("⚠️ 无法提取地址，请重新输入。")
            state["requires_clarification"] = True
            state["clarification_question"] = "抱歉，我没能理解您的地址。请提供更详细的配送地址（街道、小区、门牌号）。"
            return state    
    
    def place_order_node(self, state: AgentState) -> AgentState:
        """下单节点"""
        # 暂时以print的方式显示，后期mcp接口对接后可以改成调用接口的方式
        return state["current_order"].get_order_summary()

    def generate_response_node(self, state: AgentState) -> AgentState:
        """生成回复节点"""
        print("\n✅ 进入 generate_response_node")
        
        # 如果已经有澄清问题，直接使用
        if state.get("clarification_question"):
            response = state["clarification_question"]
            print(f"使用澄清问题: {response}")
        else:
            # 根据当前阶段生成回复
            print(f"根据阶段 '{state.get('stage')}' 和意图 '{state.get('intent')}' 生成回复")
            response = self._generate_stage_response(state)
            print(f"生成的回复: {response}")
        
        state["response"] = response
        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": state["user_input"]},
            {"role": "assistant", "content": response}
        ]
        
        return state
    
    def _generate_stage_response(self, state: AgentState) -> str: # 后续response让llm来生成todo
        """根据当前阶段生成回复 - 修复版（不调用节点函数）"""
        stage = state.get("stage", stage_enum.stage.ORDERING)
        intent = state["intent"]

        # 下单阶段
        if stage == stage_enum.stage.ORDERING:
            if intent == intentions_enum.Intentions.SEARCH and state.get("search_results"):
                return self._build_search_results(state)
            else:
                return "请问您想吃点什么？可以直接跟我说哦！"
        
        # 1. 价格确认阶段
        if stage == stage_enum.stage.CONFIRMING_PRICE:
            return self._build_price_confirmation(state)
        
        # 2. 地址确认阶段
        elif stage == stage_enum.stage.CONFIRMING_ADDRESS:
            return self._build_address_request(state)
        
        # 3. 下单阶段
        elif stage == stage_enum.stage.PLACING_ORDER:
            return self._build_order_confirmation(state)
        
        # 4. 已完成
        elif stage == stage_enum.stage.COMPLETED:
            return "订单已确认！预计30-40分钟送达。感谢您的来电！"
        
        # 5. 搜索结果
        elif intent == intentions_enum.Intentions.SEARCH and state.get("search_results"):
            return self._build_search_results(state)
        
        # 6. 问候
        elif intent == intentions_enum.Intentions.GREETING:
            return "您好！欢迎致电美味餐厅，我是AI线上外卖客服。请问今天想吃点什么？"
        
        # 7. 告别
        elif intent == intentions_enum.Intentions.FAREWELL:
            return "感谢您的光临，祝您用餐愉快！再见！"
        
        # 8. 帮助
        elif intent == intentions_enum.Intentions.HELP:
            return self._build_help()
        
        # 9. 默认
        else:
            return "我是AI线上外卖客服，可以帮您点餐、查询菜单、处理订单。请问有什么需要？"
    
    def _build_price_confirmation(self, state: AgentState) -> str:
        """构建价格确认回复"""
        order_obj = state.get("current_order")
        
        if not order_obj or len(order_obj.order_data["items"]) == 0:
            return "你还没有下单哦，您想点什么可以直接跟我说哦，或者询问我菜单也可以的!"
        
        items_str = "你共有以下菜品在订单中：\n"
        for item_name, quantity in order_obj.order_data["items"].items():
            temp_str = f"{quantity}个{item_name},"
            items_str += temp_str

        items_str += f"\n总价是{order_obj.order_data['total_price']}元。"
        items_str += "请您确认价格，如果没问题请说确认。还需要点其他菜品吗？"

        return items_str
    
    def _build_address_request(self, state: AgentState) -> str:
        """构建地址请求回复"""
        customer_info = state.get("customer_info", {})
        
        if not customer_info.get("address"):
            return "请提供您的配送地址（街道、小区、门牌号）"
        elif not customer_info.get("phone"):
            return f"地址已记录：{customer_info['address']}\n请留下您的联系电话"
        else:
            return f"地址：{customer_info['address']}\n电话：{customer_info['phone']}\n信息确认无误，现在为您下单？"
    
    def _build_order_confirmation(self, state: AgentState) -> str:
        """构建订单确认回复"""
        order_obj = state.get("current_order")
        
        if not order_obj:
            print( "订单不存在。")
            # 开始创建订单流程
            
            return "订单不存在。"
        
        items_str = "\n".join([
            f"  • {item_name} x{quantity} = {self.menu.get_price(item_name)*quantity}元"
            for item_name, quantity in order_obj.order_data["items"].items()
        ])
        subtotal = state["order_total"]
        total = subtotal + 5
        
        return f"""订单确认：
            {items_str}
            小计：{subtotal}元
            配送费：5元
            总计：{total}元

            配送至：{state['customer_info'].get('address', '未提供')}
            联系电话：{state['customer_info'].get('phone', '未提供')}

            订单已提交！预计30-40分钟送达。感谢您的美味餐厅！"""
    
    def _build_search_results(self, state: AgentState) -> str:
        """构建搜索结果回复"""
        results = state["search_results"][:3]
        
        response = "为您找到以下菜品：\n"
        for i, r in enumerate(results, 1):
            spicy = "🌶️" * (["不辣", "微辣", "中辣", "重辣"].index(r['spicy_level']) + 1 
                           if r['spicy_level'] in ["不辣", "微辣", "中辣", "重辣"] else 0)
            response += f"\n{i}. {r['name']} {r['price']}元 {spicy}\n   {r['description'][:30]}..."
        
        response += "\n\n您可以直接说'来份[菜名]'点餐。"
        return response
    
    def _build_help(self) -> str:
        """构建帮助信息"""

        return """我是AI客服小美，可以帮您：
        • 点餐：说"来份麻婆豆腐"
        • 查询：说"辣的菜"或"推荐"
        • 订单：说"看看点了什么"
        • 结账：说"点完了"

        有什么可以帮您？"""
    
    def _save_order(self, order: Dict):
        """保存订单"""
        os.makedirs("./data/orders", exist_ok=True)
        filename = f"./data/orders/order_{order['order_id']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(order, f, ensure_ascii=False, indent=2)
        
        print(f"💾 订单已保存: {filename}")
    
    def route_by_intent(self, state: AgentState) -> str:
        """根据意图路由"""
        
        # 如果需要澄清，优先处理
        if state.get("requires_clarification"):
            result = "handle_clarification"
            print(f"  ➜ 返回: {result}")
            return result
        
        # 如果当前有明确的阶段，优先处理
        stage = state.get("stage", stage_enum.stage.ORDERING)
        stage_to_node = {
            stage_enum.stage.CONFIRMING_PRICE: "confirm_price",
            stage_enum.stage.CONFIRMING_ADDRESS: "confirm_address",
            stage_enum.stage.PLACING_ORDER: "place_order"
        }
        if stage in stage_to_node:
            result = stage_to_node[stage]
            print(f"  ➜ 返回 (阶段匹配): {result}")
            return result
        
        # 根据意图路由
        intent = state["intent"]
        intent_to_node = {
            intentions_enum.Intentions.SEARCH.value: "search_menu",           # ✅ 改为 "search_menu"
            intentions_enum.Intentions.ORDER.value: "process_order",          # ✅ 改为 "process_order"
            intentions_enum.Intentions.MODIFY_ORDER.value: "process_order",   # ✅ 改为 "process_order"
            intentions_enum.Intentions.QUERY_ORDER.value: "process_order",    # ✅ 改为 "process_order"
            intentions_enum.Intentions.CONFIRM_PRICE.value: "confirm_price",
            intentions_enum.Intentions.CONFIRM_ADDRESS.value: "confirm_address",
            intentions_enum.Intentions.PLACE_ORDER.value: "place_order",
            intentions_enum.Intentions.GREETING.value: "generate_response",
            intentions_enum.Intentions.FAREWELL.value: "generate_response",
            intentions_enum.Intentions.HELP.value: "generate_response"
        }
        
        result = intent_to_node.get(intent.value, "generate_response")
        print(f"  ➜ 返回 (意图匹配): {result}")
        return result
    
    # ============ 主接口 ============
    
    def process(self, user_input: str) -> str:
        """处理用户输入"""
        
        # 初始化状态
        initial_state = {
            "messages": [],
            "user_input": user_input,
            "intent_result": {},
            "intent": "",
            "intent_confidence": 0.0,
            "confidence": 0.0,
            "entities": {},
            "rewritten_query": "",
            "search_results": [],
            "selected_dish": None,
            "current_order": None,  # 使用 None 而不是 []
            "order_total": 0.0,
            "order_confirmed": False,
            "customer_info": {
                "name": None,
                "phone": None,
                "address": None
            },
            "stage": stage_enum.stage.ORDERING,
            "requires_clarification": False,
            "clarification_question": None,
            "response": ""
        }
        
        # 运行图
        print("graph.invoke1")
        final_state = self.graph.invoke(initial_state)
        print("graph.invoke")

        return final_state["response"]
    
    def print_help(self):
        """打印帮助信息"""
        help_text = """
            📋 可用命令：
            • 点餐： "来份麻婆豆腐"、"我要宫保鸡丁"
            • 查询： "辣的菜"、"便宜的菜"、"推荐"
            • 订单： "看看点了什么"、"我的订单"
            • 修改： "不要麻婆豆腐"、"换成鱼香肉丝"
            • 地址： "送到朝阳路1号"、"地址是..."
            • 电话： "电话13800138000"、"手机号..."
            • 结账： "点完了"、"结账"、"买单"
            • 帮助： "帮助"、"help"、"?"
            • 退出： "退出"、"quit"、"exit"
            """
        print(help_text)

    def save_conversation(self):
        """保存对话历史"""
        os.makedirs("./data/conversations", exist_ok=True)
        filename = f"./data/conversations/session_{self.session_id}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"💾 对话已保存: {filename}")
        except Exception as e:
            print(f"⚠️ 保存对话失败: {e}")    
    
    def run(self):
        """主循环 - 在这里持续对话"""
        
        
        # 显示初始问候
        initial_greeting = "您好！欢迎致电美味餐厅，我是AI客服小美。请问今天想吃什么？"
        print(f"\n🤖 客服: {initial_greeting}")
        
        # 记录初始对话
        self.conversation_history.append({
            "role": "assistant",
            "content": initial_greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        # 循环对话
        while True:
            try:
                # 获取用户输入
                user_input = input("\n👤 您: ").strip()
                
                if not user_input:
                    continue
                
                # 记录用户输入
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 检查退出命令
                if user_input.lower() in ["退出", "quit", "exit", "q"]:
                    print("\n🤖 客服: 感谢您的光临，再见！")
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": "感谢您的光临，再见！",
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                
                # 检查帮助命令
                if user_input.lower() in ["帮助", "help", "?"]:
                    self.print_help()
                    continue
                
                
                # 核心处理：调用 Agent 处理用户输入
                response = self.process(user_input)
                print(response)
                
                # 记录助手回复
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # 每5轮对话自动保存一次
                if len(self.conversation_history) % 10 == 0:
                    self.save_conversation()
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，正在保存对话...")
                self.save_conversation()
                print("感谢使用，再见！")
                break
                
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                #print("请重试或输入'帮助'查看使用说明")

                # traceback.print_exc()
