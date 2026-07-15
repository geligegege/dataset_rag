import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1. 进入节点的日志输出【节点+核心参数】
            记录任务状态【哪个任务开始了】，给前端推送进度。

        2. 参数校验，确保 state 中包含必要的输入 (如 local_file_path)，并且格式正确。
            local_dir没有传入输出文件，创建一个临时。

        3. 解析输入文件类型（如 PDF、Markdown）并设置相应的标志位 (is_pdf_read_enabled, is_md_read_enabled) 以指导后续节点的处理逻辑。
            md_path=local_file_path | pdf_path=local_file_path
            file_title=文件名

        4. 结束节点时的日志输出【节点+核心参数】，给前端推送进度。

    """
    #1. 进入节点的日志输出【节点+核心参数】
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
    add_running_task(state["task_id"], function_name)

    #2. 参数校验，确保 state 中包含必要的输入 (如 local_file_path)，并且格式正确。
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.error(f"缺少必要参数: local_file_path")
        return state

    #3、判定并且完成state属性赋值
    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
        state["file_title"] = local_file_path.split("/")[-1].replace(".pdf", "")
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
        state["file_title"] = local_file_path.split("/")[-1].replace(".md", "")
    else:
        logger.error(f"不支持的文件类型: {local_file_path}")

    #提取文件名作为标题，为了后期大模型没有识别出标题时，能有个默认标题。比如 PDF 转 Markdown 后，Markdown 没有标题了，这时候就用文件名作为标题。 

    #4. 结束节点时的日志输出【节点+核心参数】
    logger.info(f">>> [Stub] 执行节点: {function_name}, 输入参数: {state}")
    add_done_task(state["task_id"], function_name)

    return state