# HyDE节点
import sys
import time
from unittest import result

from langchain_core.messages import HumanMessage

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def step_1_create_hyde_doc(rewritten_query):
    """
    步骤1：根据重写问题生成假设性答案（HyDE文档）
    输入：重写问题文本
    输出：生成的HyDE文档文本（假设性答案）

    这里的实现可以调用LLM接口，传入重写问题，让模型生成一个假设性的答案文本。
    这个答案不需要完全准确，只要与问题相关即可，目的是通过这个假设性答案去向量化并检索相关内容，提高召回率。
    """
    llm=get_llm_client()
    #加载提示词
    hyde_prompt = load_prompt("hyde_prompt",rewritten_query=rewritten_query)

    message = [
        HumanMessage(content=hyde_prompt)
    ]

    response=llm.invoke(message)
    hyde_doc=response.content.strip()
    logger.info(f"生成的假设性答案: {hyde_doc}\n问题是: {rewritten_query}")
    return hyde_doc

def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names):
    """
    步骤2：对HyDE文档进行向量化并检索相关内容
    输入：HyDE文档文本，item_names列表
    输出：检索到的embedding_chunks列表

    这里的实现与node_search_embedding节点类似，只不过查询条件是根据hyde_doc生成的向量，以及item_names进行过滤。
    通过这个步骤，我们可以获取到与假设性答案相关的内容片段，从而提高后续处理的效果。
    """
    query=rewritten_query+hyde_doc
    embeddings = generate_embeddings([query])
    item_name_str= ', '.join(f'"{item}"' for item in item_names)
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]"
    )
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content","file_title", "title", "parent_title", "item_name"]
    )

    """
       [
       
        [
            {id ,
            distance，
            entity:
               {
                  "chunk_id", "content","file_title", "title", "parent_title", "item_name"
               }
            }   
        ]
       ]
    """   
    result=resp[0] if resp else []
    logger.info(f"HyDE节点检索到的结果: {resp}")
    return result
    

def node_search_embedding_hyde(state):
    """
    假设性答案：问题->LLM生成假设性答案->对假设性答案进行向量化->向量数据库检索->返回结果
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1. 先从state获取参数数据
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")

    # 2. 让LLM根据重写问题生成假设性答案（HyDE）
    hyde_doc=step_1_create_hyde_doc(rewritten_query)

    # 3. 进行向量数据库的混合查询，获取embedding_chunks
    resp=step_2_search_embedding_hyde(rewritten_query, hyde_doc,item_names)

    # 4. 处理查询结果赋值 hyde_embedding_chunks 属性即可


    # 搜索假设性答案
    print("搜索架设性答案！！")

    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks": resp}

if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print(result)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")