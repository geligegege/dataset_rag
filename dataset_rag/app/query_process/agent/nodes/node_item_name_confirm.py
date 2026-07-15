import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from urllib import response
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit
from sympy import Trace

from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger
from app.conf.milvus_config import milvus_config

load_dotenv(find_dotenv())

def step_3_llm_item_name_and_rewrite(original_query:str, history_chats:List[Dict[str, Any]])->Dict[str, Any]:
    '''
    根据历史记录识别item_name，重写用户问题
    :params original_query: 用户的原始查询文本
    :params history_chats: 历史会话列表，包含用户之前的提问和系统的回答，供模型参考
    :return: {
        "item_names": List[str], 提取出的核心商品名称列表，供后续节点使用
        "rewritten_query": str 重写后的查询文本，供后续节点使用}
    '''
    try:
        # 1、构建提示词
        history_text = ""
        for chat in history_chats:
            history_text += f"聊天角色：{chat['role']}，回答内容： {chat['text']}，重写问题： {chat['rewritten_query']}，关联主体： {','.join(chat['item_names'])},时间： {chat['ts']}\n"
        prompt=load_prompt("rewritten_query_and_itemnames",history_text=history_text,query=original_query)
        # 2、调用大模型进行重写和item_name提取
        llm_client = get_llm_client(json_mode=True)
        messages=[
            HumanMessage(content=prompt)
        ]
        response = llm_client.invoke(messages)
        content=response.content
        # 3、解析模型返回结果
        dict_content=json.loads(content)
        if "item_names" not in dict_content :
            dict_content["item_names"]=[]
        if "rewritten_query" not in dict_content:
            dict_content["rewritten_query"]=original_query
    except Exception as e:
        logger.error(f"调用大模型进行重写和item_name提取发生异常，错误信息：{e}")
        dict_content={
            "item_names": [],
            "rewritten_query": original_query
        }        
    logger.info(f"step_3_llm_item_name_and_rewrite, original_query={original_query}, history_chats={history_chats}, rewritten_query={dict_content['rewritten_query']}, item_names={dict_content['item_names']}")
    return dict_content

def step_4_query_milvus_item_names(item_names:List[str])->List[Dict[str, Any]]:
    '''
    根据item_name向milvus向量数据库检索相关商品信息
    :params item_names: 核心商品名称列表
    :return: [{extracted:模型item_name,matches:[{item_name:milvus中匹配到的商品名称,score:匹配分数},...]},
                {extracted:模型item_name,matches:[{item_name:milvus中匹配到的商品名称,score:匹配分数},...]}]
    '''
    final_results=[]
    #1、获取milvus客户端
    milvus_client = get_milvus_client()

    #2、获取向量稀疏和稠密
    embeddings = generate_embeddings(item_names)

    #3、执行混合搜索
    for index, item_name in enumerate(item_names):
        #1、获取当前item_name的稀疏和稠密向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        #2、拼接成混合搜索请求AnnSearchRequest对象
        reqs = create_hybrid_search_requests(dense_vector=dense_vector, sparse_vector=sparse_vector)
        #3、执行搜索，获取匹配结果
        response=hybrid_search(
            client=milvus_client, 
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=[0.8, 0.2], #稠密和稀疏向量的权重分配，可根据实际情况调整
            norm_score=True #是否对匹配分数进行归一化处理，便于不同item_name之间的比较
            )
        #4、解析搜索结果，提取匹配的商品名称和分数
        matches=[]
        if response and len(response)>0:
            for res in response[0]:
                entity=res.get("entity","{}")
                hit_name=entity.get("item_name","")
                score=res.get("distance",0)
                if item_name:
                    matches.append({"item_name": hit_name, "score": score})
        final_results.append({"extracted": item_name, "matches": matches})            
    logger.info(f"step_4_query_milvus_item_names, item_names={item_names}, final_results={final_results}")
    #4、返回最终结果
    return final_results

def step_5_classify_item_names(query_milvus_results:List[Dict[str, Any]])->Dict[str, Any]:
    '''
    根据milvus的查询结果对item_name进行分类，确定的（score>0.8）直接使用，模糊的（0.5<score<=0.8）需要人工确认，其他的直接丢弃
    :params query_milvus_results: step_4_query_milvus_item_names的输出结果
    [{extracted:模型item_name,matches:[{item_name:milvus中匹配到的商品名称,score:匹配分数},...]},
    {extracted:模型item_name,matches:[{item_name:milvus中匹配到的商品名称,score:匹配分数},...]}]
    :return: {
        "confirmed_item_names": List[str], 确定的商品名称列表，供后续节点使用
        "options_item_names": List[str], 模糊的商品名称列表，需要人工确认后续使用
    }
    评分规则：
    - score > 0.85：高度匹配，直接使用
    - 0.6 < score <= 0.85：中度匹配，作为选项提供人工确认
    - score <= 0.6：低度匹配，丢弃
    '''
    #1、准备两个列表分别存储确定的和模糊的商品名称
    confirmed_item_names=[]
    options_item_names=[]
    #2、遍历milvus查询结果，根据评分规则进行分类
    for meta in query_milvus_results:
        #3、进行分数排序（倒序），提取0.85以上的作为确定，0.6-0.85的作为选项，其他丢弃
        meta["matches"].sort(key=lambda x: x["score"], reverse=True)
        high_score_matches = [m for m in meta["matches"] if m["score"] >= 0.85]
        mid_score_matches = [m for m in meta["matches"] if 0.6 < m["score"]]
        logger.info(f"step_5_classify_item_names, extracted={meta['extracted']}, 高分匹配：{high_score_matches}, 模糊分数：{mid_score_matches}")
        #4、处理高分
        if len(high_score_matches)==1:
            confirmed_item_names.append(high_score_matches[0]["item_name"])
            continue
        if len(high_score_matches)>1:
            #优先考虑名字完全匹配的
            same_name_item=None
            for item in high_score_matches:
                if item["item_name"] == meta["extracted"]:
                    same_name_item = item
                    break
            if not same_name_item:
                same_name_item=high_score_matches[0]
            confirmed_item_names.append(same_name_item["item_name"])
            continue
        #5、处理中分
        if len(mid_score_matches)>0:
            for item in mid_score_matches[:2]: #只保留前2个作为选项，避免选项过多导致人工确认困难
                options_item_names.append(item["item_name"])
            continue    
        logger.info(f"没有找到合适的匹配，丢弃该item_name，extracted={meta['extracted']}, matches={meta['matches']}")        

    #6、记录日志，返回结果
    result={
        "confirmed_item_names": list(set(confirmed_item_names)), #去重，避免重复的商品名称
        "options_item_names": list(set(options_item_names)) #去重，避免重复的商品名称
    }
    logger.info(f"step_5_classify_item_names, 处理结果: {result}")
    return result

def step_6_deal_list(state, list_results, history_chats, rewritten_query):
    """
    根据集合中的数据判断是否要赋值answer
    :params state: 当前状态
    :params list_results: step_5_classify_item_names的输出结果
    :params history_chats: 历史会话列表，包含用户之前的提问和系统的回答，供模型参考
    :return: answer: str 最终返回给用户的答案文本
    """
    #1、先获取两个集合（confirmed_item_names和options_item_names）
    confirmed_item_names = list_results.get("confirmed_item_names", [])
    options_item_names = list_results.get("options_item_names", [])
    #2、确认confirmed_item_names集合有数据（处理）
    if len(confirmed_item_names)>0:
        #更新state中的item_names为confirmed_item_names，供后续节点使用
        state["item_names"]=confirmed_item_names
        state["rewritten_query"]=rewritten_query
        state["history"]=history_chats
        if "answer" in state:
            del state["answer"]
        logger.info(f"确认的商品名称：{confirmed_item_names}，已更新状态并继续后续处理，state={state}")    
        return state    
    #3、确认options_item_names集合有数据（处理）
    if len(options_item_names)>0:
        option_names="、".join(options_item_names)
        answer=f"根据您的提问，我们不太确定您是想查询 {option_names} 中的哪一个商品。请您确认一下具体是哪个商品，我们才能更准确地为您提供帮助。"
        state["answer"]=answer
        logger.info(f"模糊的商品名称：{options_item_names}，已更新状态并等待人工确认，state={state}")
        return state
    #4、都没有
    answer="很抱歉，我们无法从您的提问中识别出相关的商品信息。请您尝试重新描述一下您的问题，或者提供更多的细节，我们会尽力帮助您解决问题。"
    state["answer"]=answer
    logger.info(f"没有匹配的商品名称，已更新状态并等待用户重新提问，state={state}")
    return state

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    #核心目标：
       1、提取【item_name】（大模型从历史会话+本次提问-》item_name-》向量库搜素->打分->abc）
       2、利用模型重写用户的问题，确保后续查询召回率更高！！！
    参数：
    - original_query: 用户的原始查询文本||session_id
    - item_names: 提取出的核心商品名称列表，供后续节点使用
    - rewritten_query: 重写后的查询文本，供后续节点使用
    - history: 历史会话列表，包含用户之前的提问和系统的回答，供模型参考
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    # 1、获取历史聊天记录
    history_chats=get_recent_messages(state["session_id"], limit=10)
    # 2、保存当前用户提问到历史记录中，供后续节点使用

    # 3、调用大模型进行重写
    """
    目的是：1、消除指代不清，如它 他 她，明确item主体
        2、重写问题，提升后续检索的召回率
        3、去掉多余的修饰词和口语，突出核心item_name
        4、补全上下文信息，如用户之前提过类似问题，这次又提了什么，结合起来更好理解用户意图
    """
    item_name_and_rewritten_query=step_3_llm_item_name_and_rewrite(state["original_query"],history_chats)
    # 4、milvus向量数据库检索，获取相关商品信息（可选，视情况而定）
    item_names = item_name_and_rewritten_query["item_names"]
    rewritten_query = item_name_and_rewritten_query["rewritten_query"]
    item_results = {"confirmed_item_names": [], "options_item_names": []}
    if len(item_names)>0:
        #查询到的item_name更新到历史记录中，供后续节点使用
        query_milvus_results=step_4_query_milvus_item_names(item_names)
        #5、查询结果进行处理区分，确定的item_name（score>0.8）直接使用，模糊的（0.5<score<=0.8）需要人工确认，其他的直接丢弃
        item_results=step_5_classify_item_names(query_milvus_results)
    #6、根据处理结果判断是否要赋值answer，是否需要人工确认
    state=step_6_deal_list(state,item_results,history_chats,rewritten_query)
    # 7、记录本次聊天对话（answer）
    logger.info(f"调试信息, session_id={state['session_id']}, original_query={state['original_query']}, rewritten_query={state.get('rewritten_query','')}, item_names={state.get('item_names',[])}, answer={state.get('answer','')}")
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query",""),
        item_names=state.get("item_names",[]),
        image_urls=[]
    )
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为擎云W585",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")
