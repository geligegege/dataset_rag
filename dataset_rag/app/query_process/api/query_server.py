#6个接口： 健康状态 返回页面 发起提问 sse长连接 查看历史对话 清空历史对话
from operator import is_
from pathlib import Path
from urllib import response
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT
# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    logger.info("出发后台检测，目前一切正常！")
    return {"status": "ok"}

#返回chat.html页面
@app.get("/chat.html")
async def get_chat_page():

    chat_htmml_path=PROJECT_ROOT/"app/query_process/page/chat.html"
    if not chat_htmml_path.exists():
        logger.error(f"chat页面文件不存在，路径：{chat_htmml_path}")
        raise HTTPException(status_code=404, detail="chat页面文件不存在")
    return FileResponse(path=chat_htmml_path)

#发起提问接口
#接受参数的类型
class QueryRequest(BaseModel):
    session_id: str = Field(..., title="会话ID，前端生成的唯一标识")#必须传递
    query: str = Field(..., title="用户的查询内容")
    is_stream: bool = Field(False, title="是否使用SSE流式返回结果，默认为False")

def run_query_graph(session_id:str, query:str, is_stream:bool):
    update_task_status(session_id, "processing", is_stream)  # 更新状态，标记为流式处理中,is_stream=True会在后续的节点处理函数里触发数据推送到队列中
    
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)  # 查询图处理完成，更新状态为done
    except Exception as e:
        logger.exception(f"查询过程中发生错误，session_id={session_id}, error={e}")
        update_task_status(session_id, "failed", is_stream)  # 查询图处理发生错误，更新状态为error
        push_to_session(session_id, SSEEvent, {"error": str(e)})
    
@app.post("/query")    #客户端-》问题-》graph-》查rag-》返回
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    ：param request: QueryRequest - 包含 session_id（会话唯一ID）、query（用户查询内容）和 is_stream（是否使用SSE流式返回结果）
    ：param background_tasks: BackgroundTasks - FastAPI提供的后台任务工具
    ：return: StreamingResponse 
    """
    query=request.query
    session_id=request.session_id or str(uuid.uuid4())  # 如果前端没有传session_id，就生成一个新的唯一ID
    is_stream=request.is_stream

    #判断是不是流式处理 开始处理|后台运行图，结果向前端推送
    if is_stream:
        #只要开启流式处理，就创建一个SSE队列，供图的节点往里丢数据，前端从这个接口拿数据
        create_sse_queue(session_id)
        
        # 流式处理
        background_tasks.add_task(run_query_graph, session_id, query, is_stream)

        logger.info(f"已启动后台任务处理查询，session_id={session_id}, query={query}, is_stream={is_stream}")
        return{
            "session_id": session_id,
            "message": "查询中......",
        }
    else:
        # 同步运行图，等待结果返回后再响应
        run_query_graph(session_id, query, is_stream)
        #获取最后一个节点插入的结果
        answer = get_task_result(session_id,"answer")
        logger.info(f"同步查询完成，session_id={session_id}, query={query}, answer={answer}")
        return{
            "session_id": session_id,
            "answer": answer,
            "message": "查询已完成，结果已返回",
            "done_list":[]
        }
    

@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """
    SSE长连接接口，前端通过这个接口接收查询图处理过程中各个节点的输出结果。

    参数：
    - session_id: 会话ID，用于识别不同的查询任务
    - request: Request对象，用于检测客户端连接状态

    返回：
    - StreamingResponse: 以SSE格式持续推送数据给前端
    """
    logger.info(f"客户端连接到SSE流，session_id={session_id}")
    return StreamingResponse(sse_generator(session_id,request), media_type="text/event-stream")    


@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10):
    """
    查看历史对话接口，根据session_id查询历史对话记录。

    参数：
    - session_id: 会话ID，用于识别不同的查询任务
    - limit: 返回的历史记录条数，默认为10

    返回：
    - dict: 包含历史对话记录的字典
    """
    chats = get_recent_messages(session_id, limit)

    logger.info(f"查询历史对话记录，session_id={session_id}, limit={limit}")    
    return {
        "session_id": session_id,
        "items": chats
    }    


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    delete_count = clear_history(session_id)
    logger.info(f"清空历史对话记录，session_id={session_id}, 删除数量={delete_count}")
    return {
        "deleted_count": delete_count,
        "message": f"已清空会话ID {session_id} 的历史对话记录"
    }

if __name__ == "__main__":
    uvicorn.run("app.query_process.api.query_server:app", host="0.0.0.0", port=8080, reload=True)