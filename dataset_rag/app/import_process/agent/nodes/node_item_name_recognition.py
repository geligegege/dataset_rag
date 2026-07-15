from operator import index
import os
import sys
from typing import List, Dict, Any, Tuple, final

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pydantic.v1.networks import Parts
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_done_task, add_running_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.conf.milvus_config import milvus_config

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

""" 
    主要目标：
        1. 录用文本大模型识别当前chunks对应的item_name!用于区分不同的文档
        2.使用嵌入式模型，将item_name生成向量存储到向量数据
        3.修改state[chunks]->chunk{title parent_title content...}
    实现步骤：
        1.校验和取值（file_title,chunks）
        2.构建上下文环境 chunks->top 5->拼接成context文本
        3.调用文本大模型，输入context文本，输出item_name
        4.将item_name生成向量（稠密/稀疏）
        6.存储向量到向量数据库
"""

def step_1_get_chunks(state: ImportGraphState) -> Tuple[str, List[Dict[str, Any]]]:
    """
    步骤1: 校验和取值
    目标: 从 state 中获取 file_title 和 chunks，并进行基本校验。
    输入: state (ImportGraphState)
    输出: file_title (str), chunks (List[Dict])
    主要操作:
        1. 从 state 中获取 file_title 和 chunks。
        2. 校验 file_title 是否存在且为字符串。
        3. 校验 chunks 是否存在且为列表。
        4. 返回 file_title 和 chunks。
    """
    chunks = state.get("chunks")
    print(f"步骤1: 从 state 中获取 chunks，初始值: {chunks}")
    file_title = state.get("file_title")
    if not chunks or not isinstance(chunks, list):
        raise ValueError("state中缺少chunks或chunks不是列表")
    if not file_title :
        #md_path中获取文件名作为file_title
        file_title = os.path.basename(state.get("md_path"))
        state["file_title"] = file_title
        logger.warning(f"state中缺少file_title，已从md_path中提取并设置为: {file_title}")
    logger.info(f"步骤1: 获取到 file_title: {file_title}, chunks数量: {len(chunks)}")

    return file_title, chunks

def step_2_build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    步骤2: 构建上下文环境
    目标: 从 chunks 中取前5个切片，拼接成字符串，作为大模型识别商品名称的输入上下文。
    输入: chunks (List[Dict])
    输出: context (str)
    主要操作:
        1. 从 chunks 中取前 DEFAULT_ITEM_NAME_CHUNK_K 个切片,最大长度不超过 CONTEXT_TOTAL_MAX_CHARS。
        2. 将这些内容拼接成一个字符串，作为上下文返回。
    截取内容处理：
        切片：{1}.标题：{title},内容{content} \n\n  
        切片：{2}.标题：{title},内容{content} \n\n 
        切片：{3}.标题：{title},内容{content} \n\n 
        切片：{4}.标题：{title},内容{content} \n\n 
        切片：{5}.标题：{title},内容{content} \n\n 
    """
    parts=[]
    total_chars=0
    for i,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        title=chunk.get("title","")
        content=chunk.get("content","")
        part=f"切片：{i}.标题：{title},内容{content} \n\n"
        parts.append(part)
        total_chars+=len(part)
        if total_chars>=CONTEXT_TOTAL_MAX_CHARS:
            logger.warning(f"构建上下文时，已达到总字符数上限 {CONTEXT_TOTAL_MAX_CHARS}，将停止添加更多切片")
            break
    context="\n\n".join(parts)
    final_context=context[:CONTEXT_TOTAL_MAX_CHARS]
    logger.info(f"步骤2: 构建上下文环境，最终上下文字符数: {len(final_context)}")
    return final_context



def step_3_get_item_name(context: str, file_title: str) -> str:
    """
    步骤3: 调用模型，拼接提示词，获取商品名称
    目标: 使用大模型识别商品名称。
    输入: context (str), file_title (str)
    输出: item_name (str)
    主要操作:
        1. 加载识别商品名称的提示词模板。
        2. 将 context 填充到提示词中，构建完整的提示词文本。
        3. 调用大模型接口，输入提示词文本，获取模型输出。
        4. 从模型输出中提取 item_name，并进行必要的清洗和校验。
        5. 返回 item_name。
    """
    human_prompt=load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt=load_prompt("product_recognition_system")

    llm=get_llm_client(json_mode=False)
    messages=[
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    response=llm.invoke(messages)
    item_name=response.content
    if not item_name:
        raise ValueError(f"模型返回的 item_name 无效: {item_name}")
    logger.info(f"步骤3: 模型识别到的 item_name: {item_name}")
    return item_name

def step_4_update_chunks_and_state(state: ImportGraphState, item_name: str, chunks: List[Dict[str, Any]]):
    """
    步骤4: 修改 state，将 chunks 中的 item_name 字段更新为识别到的 item_name。
    目标: 将识别到的 item_name 存入 state 中，供后续节点使用。
    输入: state (ImportGraphState), item_name (str), chunks (List[Dict])
    输出: None (直接修改 state)
    主要操作:
        1. 遍历 chunks 列表，将每个切片的 item_name 字段更新为识别到的 item_name。
        2. 将更新后的 chunks 列表重新赋值回 state["chunks"]。
    """
    state["item_name"]=item_name

    for chunk in chunks:
        chunk["item_name"]=item_name
    state["chunks"]=chunks
    logger.info(f"步骤4: 已将识别到的 item_name 更新到 state 中，并赋值给每个切片")




def step_5_generate_vectors(item_name: str) -> Tuple[List[float], List[float]]:
    """
    步骤5: item_name 生成向量
    目标: 使用嵌入式模型将 item_name 转换为稠密向量和稀疏向量。
    输入: item_name (str)
    输出: dense_vector (List[float]), sparse_vector (List[float])
    主要操作:
        1. 获取 BGE-M3 模型实例。
        2. 调用 generate_embeddings 方法，输入 item_name，获取稠密向量和稀疏向量。
        3. 对稀疏向量进行归一化处理。
        4. 返回稠密向量和归一化后的稀疏向量。
    """
    logger.info(f"步骤5: 开始为 item_name 生成向量，item_name: {item_name}")
    result=generate_embeddings([item_name])#自定义方法，输入字符串列表，输出对应的稠密向量和稀疏向量列表
    dense_vector=result["dense"][0]
    sparse_vector=result["sparse"][0]
    logger.info(f"步骤5: 已生成稠密向量和稀疏向量，稠密向量长度: {len(dense_vector)}, 稀疏向量长度: {len(sparse_vector)}")
    return dense_vector, sparse_vector

def step_6_save_to_vector_db(file_title: str, item_name: str, dense_vector: List[float], sparse_vector: List[float]):

    """
    步骤6: 将向量存储到数据库 kb_item_name 中 (id/file_title/item_name/vector)
    目标: 将生成的向量数据存储到 Milvus 向量数据库中，供后续检索使用。
    输入: state (ImportGraphState), file_title (str), item_name (str), dense_vector (List[float]), sparse_vector (List[float])
    输出: None
    主要操作:
        1. 获取 Milvus 客户端实例。
        2. 构建要插入的数据格式，包括 id、file_title、item_name、dense_vector 和 sparse_vector。
        3. 调用 Milvus 客户端的插入方法，将数据写入 kb_item_name 集合。
        4. 处理插入结果，确保数据成功存储.
    """
    #1、获取Milvus客户端实例
    milvus_client=get_milvus_client()
    #2、判断是否存在集合kb_item_name，不存在则创建
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        schema=milvus_client.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        
        index_params=milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector", #给哪个列创建索引
            index_name="dense_vector",#索引的名字
            index_type="HNSW", #查找索引的办法
            metric_type="COSINE", #距离计算方式
            params={"M": 16, "efConstruction": 200} #HNSW算法的参数(一万级数据推荐M=16，efConstruction=200，数据量大可以适当调大)
        )

        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"} #稀疏向量推荐DAAT_MAXSCORE算法,只计算包含查询向量非零特征的切片，提升检索效率
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema, 
            index_params=index_params
        )

    #3、按文件标题清理旧数据，避免同一文件重复导入产生重复记录
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    escaped_item_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=milvus_config.item_name_collection,
        filter=f'item_name == "{escaped_item_name}"'
    )
    #4、插入数据
    item={
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    insert_result=milvus_client.insert(collection_name=milvus_config.item_name_collection, data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"步骤6: 已将 item_name 向量数据存储到 Milvus，插入结果: {insert_result}")     



def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}")
    add_running_task(state["task_id"], function_name)

    try:
        #1、验证和取值
        file_title,chunks=step_1_get_chunks(state)
        #2、构建上下文环境 取chunks前5个切片，拼接成字符串，作为大模型识别商品名称的输入上下文
        context = step_2_build_context(chunks)
        #3、调用模型，拼接提示词，获取商品名称
        item_name = step_3_get_item_name(context,file_title)
        #4、修改state chunks->item_name
        step_4_update_chunks_and_state(state, item_name,chunks)
        #5、item_name生成向量
        dense_vector, sparse_vector = step_5_generate_vectors(item_name)
        #6、将向量存储到数据库kb_item_name中(id/file_title/item_name/vector)
        step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)

    except Exception as e:
        #处理异常
        logger.error(f"执行节点 {function_name} milvus发生异常: {e}")
        raise

    finally:
        #6、结束日志和任务状态的配置
        logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
        add_done_task(state["task_id"], function_name)

    return state 

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # # 4. 验证Milvus存储（可选）
        # milvus_client = get_milvus_client()
        # collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        # if milvus_client and collection_name:
        #     milvus_client.load_collection(collection_name)
        #     # 检索测试结果
        #     item_name = result_state.get('item_name')
        #     safe_name = _escape_milvus_string(item_name)
        #     res = milvus_client.query(
        #         collection_name=collection_name,
        #         filter=f'item_name=="{safe_name}"',
        #         output_fields=["file_title", "item_name"]
        #     )
        #     logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()