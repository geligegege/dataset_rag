import sys
import os
from typing import Any, List, Dict

from dotenv import load_dotenv
from langsmith import expect

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph核心节点：BGE-M3文本向量化处理
    主流程（串行执行，全流程异常隔离）：
        1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
        2. 模型初始化：获取BGE-M3单例模型实例，避免重复加载
        3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
        4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
    参数：
        state: ImportGraphState - 流程全局状态对象，包含上游传入的chunks、task_id等数据
    返回：
        ImportGraphState - 更新后的状态对象，chunks字段新增dense_vector/sparse_vector
    异常处理：
        节点内所有异常均捕获，不终止整体LangGraph流程，仅记录错误日志
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        # 1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
        chunks = state.get("chunks", [])
        if not chunks:
            logger.error("输入数据缺失：chunks列表为空，无法执行向量化处理")
            return ValueError("输入数据缺失：chunks列表为空，无法执行向量化处理")
        '''
        生成向量的时候用f'商品名：item_name,介绍:content'的格式拼接文本，给模型更多提示信息，提升向量质量。 --- IGNORE ---
        核心词前置，让模型更聚焦于商品名称和介绍内容，提升生成的稠密向量和稀疏向量的表达能力。 --- IGNORE ---
        模型的上下文窗口为8192token，按照经验公式1token≈4字符计算，文本长度不宜超过3000字符，避免输入过长导致模型性能下降。 --- IGNORE ---
        '''
        final_chunks = []
        batch_size = 5  # 根据实际情况调整批处理大小，平衡效率和内存占用
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            texts_to_embed = []
            for chunk in batch_chunks:
                item_name = chunk.get("item_name", "")
                content = chunk.get("content", "")
                # 拼接文本，给模型更多提示信息
                text = f"商品名：{item_name}, 介绍: {content}"
                texts_to_embed.append(text)


            # 3. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            embedding_result = generate_embeddings(texts_to_embed)
            dense_vectors = embedding_result.get("dense", [])
            sparse_vectors = embedding_result.get("sparse", [])

            for idx, chunk in enumerate(batch_chunks):
                if idx < len(dense_vectors):
                    chunk["dense_vector"] = dense_vectors[idx]
                if idx < len(sparse_vectors):
                    chunk["sparse_vector"] = sparse_vectors[idx]
                final_chunks.append(chunk)
        state["chunks"] = final_chunks  # 4. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用
        logger.info(f"--- 文本向量化处理完成，处理了 {len(final_chunks)} 个切片 ---")
        add_done_task(state.get("task_id", ""), current_node)        
    except Exception as e:
        logger.error(f"chunks向量化处理发生异常: {e}")
        return state  # 返回原状态，继续执行后续节点
    # 返回更新后的状态对象，传递至下游节点
    return state

# ==========================================
# 本地单元测试入口
# 功能：独立验证向量化节点全链路逻辑，无需启动整个LangGraph流程
# 适用场景：本地开发、调试、模型有效性验证
# ==========================================
if __name__ == '__main__':
    # 加载环境变量：定位项目根目录下的.env，读取模型路径/设备等配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造模拟测试状态：模拟上游节点输出的chunks数据，贴合真实业务场景
    test_state = ImportGraphState({
        "task_id": "test_task_embedding_001",  # 测试任务ID
        "chunks": [  # 模拟带item_name的文本切片（上游商品名称识别节点产出）
            {
                "content": "这是一个测试文档的内容，用于验证向量化是否成功。",
                "title": "测试文档标题",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            },
            {
                "content": "这是第二个测试文档的内容，用于验证批量处理逻辑。",
                "title": "测试文档标题2",
                "item_name": "测试项目",
                "file_title": "测试文件.pdf"
            }
        ]
    })

    # 执行本地测试
    logger.info("=== BGE-M3向量化节点本地单元测试启动 ===")
    try:
        # 调用核心节点函数
        result_state = node_bge_embedding(test_state)
        # 提取测试结果
        result_chunks = result_state.get("chunks", [])

        # 打印测试结果统计
        logger.info(f"=== 向量化节点本地测试完成 ===")
        logger.info(f"测试任务ID：{test_state.get('task_id')}")
        logger.info(f"待处理切片数：2 | 实际处理切片数：{len(result_chunks)}")

        # 验证向量生成结果（打印向量字段是否存在）
        for idx, chunk in enumerate(result_chunks):
            has_dense = "dense_vector" in chunk
            has_sparse = "sparse_vector" in chunk
            logger.info(
                f"第{idx + 1}条切片：稠密向量生成{'' if has_dense else '未'}成功 | 稀疏向量生成{'' if has_sparse else '未'}成功")

    except Exception as e:
        logger.error(f"=== 向量化节点本地测试失败 ===" f"错误原因：{str(e)}", exc_info=True)
        # 新手友好提示：给出核心排查方向
        logger.warning("排查提示：请检查BGE-M3模型路径、显存是否充足、环境变量配置是否正确")