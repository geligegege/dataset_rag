import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_done_task, add_running_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection

def step_2_prepare_collection(state: ImportGraphState):
    """
    创建Milvus集合（如果不存在），并配置索引
    """
    try:
        #1、获取Milvus客户端实例
        milvus_client=get_milvus_client()
        #2、判断是否存在集合kb_item_name，不存在则创建
        if not milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
            schema=milvus_client.create_schema(
                auto_id=True,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
            schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="part", datatype=DataType.INT8)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            
            index_params=milvus_client.prepare_index_params()

            index_params.add_index(
                field_name="dense_vector", #给哪个列创建索引
                index_name="dense_vector",#索引的名字
                index_type="HNSW", #查找索引的办法
                metric_type="COSINE", #距离计算方式
                params={"M": 32, "efConstruction": 300} #HNSW算法的参数(一万级数据推荐M=16，efConstruction=200，数据量大可以适当调大)
            )

            index_params.add_index(
                field_name="sparse_vector",
                index_name="sparse_vector",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="IP",
                params={"inverted_index_algo": "DAAT_MAXSCORE"} #稀疏向量推荐DAAT_MAXSCORE算法,只计算包含查询向量非零特征的切片，提升检索效率
            )
            milvus_client.create_collection(
                collection_name=milvus_config.chunks_collection,
                schema=schema, 
                index_params=index_params
            )
    except Exception as e:
        logger.error(f"准备Milvus集合发生异常: {e}")
        raise
    return milvus_client

def step_3_delete_old_data(state: ImportGraphState, milvus_client):
    """
    根据商品名称删除旧数据，避免同一文件重复导入产生重复记录
    """

    item_name = state.get("item_name", "")
    if not item_name:
        logger.info("状态数据缺失：item_name为空，不执行旧数据删除")
        return
    escaped_item_name = escape_milvus_string(item_name)
    milvus_client.delete(
        collection_name=milvus_config.chunks_collection,
        filter=f'item_name == "{escaped_item_name}"'
    )


def step_4_insert_collections(chunks: List[Dict[str, Any]], milvus_client):
    """
    批量插入数据到Milvus集合
    """
    insert_result=milvus_client.insert(
        collection_name=CHUNKS_COLLECTION_NAME,
        data=chunks
    )
    insert_count=insert_result.get("insert_count", 0)
    logger.info(f"成功插入 {insert_count} 条数据到Milvus集合")

    ids=insert_result.get("ids", [])
    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]

    return chunks        

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """

    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}")
    add_running_task(state["task_id"], function_name)

    try:
        #1、验证和取值
        chunks = state.get("chunks", [])
        if not chunks:
            logger.error("输入数据缺失：chunks列表为空，无法执行Milvus导入")
            raise ValueError("输入数据缺失：chunks列表为空，无法执行Milvus导入")
        #2、创建集合
        milvus_client=step_2_prepare_collection(state)
        #3、按商品名称清理旧数据，避免同一文件重复导入产生重复记录
        step_3_delete_old_data(state, milvus_client)
        #4、批量插入数据
        with_id_chunks = step_4_insert_collections(chunks,milvus_client)
        state["chunks"] = with_id_chunks  # 将插入结果（包含chunk_id）更新回状态对象，供后续节点使用


    except Exception as e:
        #处理异常
        logger.error(f"执行节点 {function_name} mineru解析发生异常: {e}")
        raise

    finally:
        #6、结束日志和任务状态的配置
        logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
        add_done_task(state["task_id"], function_name)

    return state 

if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":2,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")