import sys
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.task_utils import add_running_task

load_dotenv()
# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5 # 最大间断分值

def step_1_merge_rrf_mcp(state):
    """
    步骤1：合并 RRF 和 MCP 的结果，形成统一的候选列表
    输入：state 中的 RRF 结果和 MCP 结果
    输出：合并后的候选列表，每个候选包含必要的字段（如 text、chunk_id、title、url、source 等）

    这里的实现需要根据 RRF 和 MCP 的输出格式进行适配，将它们合并成一个统一的列表，方便后续的 Rerank 处理。
    合并时可以考虑一些简单的规则，比如如果 MCP 有结果就用 MCP 的文本和标题，否则用 RRF 的内容；URL 只有 MCP 有等。
    """
    rrf_results = state.get("rrf_chunks", [])
    mcp_results = state.get("web_search_docs", [])
    # 合并逻辑
    merged_results = []
    for chunk in rrf_results:
        merged_results.append({
            "text": chunk.get("entity", {}).get("content", ""),
            "chunk_id": chunk.get("id"),
            "title": chunk.get("entity", {}).get("title", ""),
            "url": None,
            "source": "local"
        })
    for doc in mcp_results:
        merged_results.append({
            "text": doc.get("snippet", ""),
            "chunk_id": None,
            "title": doc.get("title", ""),
            "url": doc.get("url"),
            "source": "web"
        })
    logger.info(f"完成了RRF和MCP结果的合并，合并后的候选列表长度为：{len(merged_results)}")
    return merged_results


def step_2_rerank_doc_list(doc_list,state):
    """
    步骤2：使用 Cross-Encoder 模型对合并后的候选列表进行精确打分重排
    输入：合并后的候选列表和原始查询（从 state 中获取）
    输出：带有 Rerank 分数的候选列表

    这里需要加载预训练的 Cross-Encoder 模型，对每个候选与查询进行打分，得到一个新的 score 字段。可以使用模型的 batch 处理能力来加速评分过程。
    """
    query = state.get("rewritten_query", "") or state.get("original_query", "")
    reranker_model = get_reranker_model()
    # 准备输入数据
    texts = [doc["text"] for doc in doc_list]
    question_pairs = [[query, text] for text in texts]
    # 批量打分
    scores = reranker_model.compute_score(question_pairs,normalize=True)#本次模型的上下文长度限制在512（一个pair）,多余的token会被截断
    # 将分数添加到 doc_list 中
    for doc, score in zip(doc_list, scores):
        doc["score"] = score
    # 排序
    doc_list.sort(key=lambda x:x['score'],reverse=True)        
    logger.info(f"完成了Rerank打分处理完毕，样例结果为：{doc_list[:3]}")
    return doc_list


def step_3_topk_and_gap(rerank_score_list):
    """
    步骤3：根据 Rerank 分数进行 TopK 和断崖处理
    输入：带有 Rerank 分数的候选列表
    输出：最终的精排结果列表

    这里需要根据预设的 TopK 上限和下限，以及分数断崖的相对和绝对阈值，对排序后的候选列表进行筛选，得到最终的输出结果。
    """
    max_topk=RERANK_MAX_TOPK    #至多获取的元素的数量
    min_topk=RERANK_MIN_TOPK    #至少获取的元素的数量
    gap_ratio=RERANK_GAP_RATIO  #断崖的百分比
    gap_abs=RERANK_GAP_ABS      #断崖的绝对分数差
    # 先取 TopK 上限的候选
    topk=min(max_topk,len(rerank_score_list))
    if topk>min_topk:
        for index in range(min_topk,topk):
            prev_score=rerank_score_list[index-1]['score']
            curr_score=rerank_score_list[index]['score']
            # 判断是否满足断崖条件
            gap = prev_score - curr_score
            gap_percent = gap / abs(prev_score)+1e-8  # 避免除零
            if gap_percent >= gap_ratio or gap >= gap_abs:
                logger.info(f"在TopK={index}处发现断崖,结束循环")
                topk=index
                break
    topk_doc_list=rerank_score_list[:topk]
    logger.info(f"完成了TopK和断崖处理，最终保留的文档数量为：{len(topk_doc_list)}")
    return topk_doc_list

def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    """
      [
         rrf = {id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
         mcp = {snippet: 内容,title:标题,url:关联的文章或者图片的地址}
         {
            text:内容 snippet content,
            chunk_id: chunk_id rrf有  mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp  || local -> rrf 
         }
      ]
    
    """
    #1、结果合并
    doc_list=step_1_merge_rrf_mcp(state)
    #2、启动rerank精排
    """
      [
         rrf = {id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
         mcp = {snippet: 内容,title:标题,url:关联的文章或者图片的地址}
         {
            text:内容 snippet content,
            chunk_id: chunk_id rrf有  mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp  || local -> rrf 
            score: 0.8  0.6  0.9
         }
      ]
    """    
    rerank_score_list=step_2_rerank_doc_list(doc_list,state)
    # 3. 启动算法进行放断崖以及topk处理  0.9  0.89  0.35
    final_doc_list = step_3_topk_and_gap(rerank_score_list)

    #4、结果赋值 state["reranked_docs"] = [{chunk}, {chunk}, {chunk}]
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": final_doc_list}              



if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")    