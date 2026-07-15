import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具


# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)

# 8080/import->import.html
@app.get("/import",response_class=FileResponse)
async def get_import_page():
    import_html_path = PROJECT_ROOT/"app/import_process/page/import.html"
    if not import_html_path.exists():
        logger.error(f"导入页面文件不存在，路径：{import_html_path}")
        raise HTTPException(status_code=404, detail="导入页面文件不存在")
    return FileResponse(path=import_html_path,media_type="text/html")


#定义调用LangGraph图的函数
def run_import_graph(task_id:str, local_file_path:str, local_dir:str):
    """
    param task_id: str - 任务唯一ID，用于日志追踪和前端进度展示
    param local_file_path: str - 文件的地址
    param local_dir: str - 输出文件夹的地址
    """
    # add_running_task(task_id, "upload_file")
    # add_done_task(task_id, "upload_file")
    try:
        update_task_status(task_id, "processing")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_file_path"] = local_file_path
        init_state["local_dir"] = local_dir
        # 构造图的初始状态，传入必要的参数
        for event in kb_import_app.stream(init_state):
            for node_name,state in event.items():
                logger.info(f"节点 {node_name} 执行完成，当前状态: {state}")

        update_task_status(task_id, "completed")
        logger.info(f"任务 {task_id} 执行完成")
    except Exception as e:    
        logger.exception(f"任务 {task_id} 执行导入流程发生异常")
        update_task_status(task_id, "failed")



#8080/upload post->文件上传->开启导入流程
"""
    1、接收文件存储到output文件夹！ /output/当天的日期/uuid/文件名
    2、开启异步，import_graph图的执行
"""
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks,files: List[UploadFile] = File(...)):
    today_str = datetime.now().strftime("%Y%m%d")
    base_out_path = PROJECT_ROOT / "output" / today_str 
    task_ids = []
    for file in files:
        task_id=str(uuid.uuid4())
        task_ids.append(task_id)

        #记录下进行文件上传了
        add_running_task(task_id, "upload_file")
        dir_path = base_out_path / task_id
        local_file_path = dir_path / file.filename#构造本地文件路径：output/当天日期/uuid/文件名
        dir_path.mkdir(parents=True, exist_ok=True)  # 创建目录（如果不存在）
        with open(local_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        

        background_tasks.add_task(run_import_graph, task_id, str(local_file_path), str(dir_path))
        logger.info(f"文件 {file.filename} 上传成功，存储路径：{local_file_path}，已添加导入任务 {task_id} 到后台执行")
        add_done_task(task_id, "upload_file")
    return {
        "code": 200,
        "message": "文件上传成功，导入任务已启动",
        "task_ids": task_ids
    }  

# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

if __name__ == "__main__":
    # 启动FastAPI应用，监听在本地8000端口，开启热重载（代码修改后自动重启服务）
    uvicorn.run(app, host="0.0.0.0", port=8080)