import sys
sys.path.append('.')  # 添加当前目录到路径

import FAISSMenuStore as faissDB
import rerankModel 
import IntentRecognizer
import llm_chater
import NLPEntityExtractor
import menu
import RestaurantAgent

vectorDB = faissDB.FAISSMenuStore()
rerankModel = rerankModel.HybridRerankStore(vectorDB)
intentRecog = IntentRecognizer.IntentRecognizer()
menus_path = "./data/menu.json"
entityExtracter = NLPEntityExtractor.NLPEntityExtractor(menus_path)
Restaurant_menu = menu.Menu(menus_path)
local_llm = llm_chater.SimpleLLM(max_history=20)
local_llm.set_system_prompt("你是一名ai线上点餐助手，请帮助用户点外卖")

agent = RestaurantAgent(Restaurant_menu, local_llm, rerankModel, intentRecog, entityExtracter)

print("模型加载完成")

agent.process("你好")