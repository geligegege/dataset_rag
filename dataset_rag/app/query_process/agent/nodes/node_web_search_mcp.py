import asyncio
import os
import json
import sys
from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp
from openai import max_retries # pip install openai-agents
from app.core.logger import  logger

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

async def call_mcp_streamable(query):
    """
    调用MCP Streamable接口，获取搜索结果并通过SSE推送给前端

    参数：
    - query: 搜索查询文本
    - session_id: 会话ID，用于识别不同的查询任务

    返回：
    - None（结果通过SSE推送给前端）
    """
    logger.info(f"调用MCP Streamable接口, query={query}")
    mcp_client = MCPServerStreamableHttp(
        name="DashScope Web Search",
        params={
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers":{"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10,
        },
        max_retry_attempts=3,
    )
    try:
            await mcp_client.connect()
            result = await mcp_client.call_tool(
                  tool_name="bailian_web_search",
                  arguments={"query": query,"count":5},
            )
            return result
    except Exception as e:
            logger.exception(f"MCP Streamable接口调用失败， error={e}")
            raise e
    finally:
            await mcp_client.cleanup()        




def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    #1、获取问题
    query = state.get("rewritten_query", "")
    # 2、调用mcp外部搜索引擎，获取结果
    results = asyncio.run(call_mcp_streamable(query))
    # 3、结果处理，赋值 state["web_search_results"] = [{title, content, url}, ...]
    # {
    #   "isError": false,
    #   "content": [
    #     {
    #       "text": "{\"pages\":[{\"snippet\":\"和讯首页|手机和讯 登录注册 股票客户端 Android 股票客户端 iPhone\",\"hostname\":\"和讯网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN01VcUfI91cc0kCH3Gt2_!!6000000003620-73-tps-32-32.ico\",
    #                               \"title\":\"行情中心-和讯网 国内全面的即时行情数据服务中心\",
    #                               \"url\":\"https://quote.hexun.com/\"},
    #                            {\"snippet\":\"数据中心\",\"hostname\":\"东方财富网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i1/O1CN01iL4mYC1cF6vgiem0A_!!6000000003570-55-tps-32-32.svg\",\"title\":\"股票\",\"url\":\"https://stock.eastmoney.com/\"},{\"snippet\":\"意见反馈\",\"hostname\":\"东方财富网\",\"hostlogo\":\"https://quote.eastmoney.com/favicon.ico\",\"title\":\"行情中心:国内快捷全面的股票、基金、期货、美股、港股、外汇、黄金、债券行情系统_东方财富网\",\"url\":\"https://quote.eastmoney.com/center/qqzs.html#!/stealingyourhistory\"}],\"request_id\":\"faa40120-ee17-4401-a6c5-9970da077c05\",\"tools\":[],\"status\":0}",
    #       "type": "text"
    #     }
    #   ]
    # }
    
    web_documents = json.loads(results.content[0].text).get("pages",[])
    # 调用mcp外部引擎
    logger.info(f"MCP Web Search节点获取到的原始结果: {web_documents}")

    print("---node-web-search-mcp处理结束---")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    return {"web_search_docs": web_documents}


from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id":"mcp_01",
        "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
        "is_stream":True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)