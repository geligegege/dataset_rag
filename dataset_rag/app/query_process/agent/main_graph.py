from langgraph.graph import StateGraph, END
from pathlib import Path

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

# 初始化状态图
builder = StateGraph(QueryGraphState)

# 注册所有节点
builder.add_node("node_item_name_confirm", node_item_name_confirm) # 确认商品
builder.add_node("node_search_embedding", node_search_embedding)   # 向量搜索
builder.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
builder.add_node("node_web_search_mcp", node_web_search_mcp)
builder.add_node("node_rrf", node_rrf)                             # 排序
builder.add_node("node_rerank", node_rerank)                       # 重排
builder.add_node("node_answer_output", node_answer_output)         # 生成

builder.set_entry_point("node_item_name_confirm")
#node_item_name_confirm可能出现没有明确主题item_name，会提前结束返回用户提示，让用户明确内容
#node_item_name_confirm->多路召回 生成答案反馈给前端（answer:str） 根据answer是否包含item_name来判断是否正确

def route_after_node_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        # 提取到回答，直接生成答案反馈给前端
        return "node_answer_output"
    else:
        # 提取到商品名称，继续进行多路搜索
        return "node_web_search_mcp","node_search_embedding","node_search_embedding_hyde"
    
builder.add_conditional_edges("node_item_name_confirm", 
                              route_after_node_item_name_confirm,
                              {
                                "node_web_search_mcp": "node_web_search_mcp",
                                "node_search_embedding": "node_search_embedding",
                                "node_search_embedding_hyde": "node_search_embedding_hyde",
                                "node_answer_output": "node_answer_output"
                              })

builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
builder.add_edge("node_rerank", "node_answer_output")
builder.add_edge("node_answer_output", END)

query_app=builder.compile()

def save_query_graph(output_dir: str | Path | None = None) -> dict[str, str]:
    """
    保存当前查询流程状态图。

    默认输出到当前 main_graph.py 同级目录，保存 Mermaid 源文件、ASCII 文本；
    如果当前环境支持 Mermaid PNG 渲染，也会额外保存 PNG 图片。
    """
    graph_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph = query_app.get_graph()
    saved_paths: dict[str, str] = {}

    mermaid_path = graph_dir / "kb_query_state_graph.mmd"
    mermaid_path.write_text(graph.draw_mermaid(), encoding="utf-8")
    saved_paths["mermaid"] = str(mermaid_path)

    try:
        ascii_path = graph_dir / "kb_query_state_graph.txt"
        ascii_path.write_text(graph.draw_ascii(), encoding="utf-8")
        saved_paths["ascii"] = str(ascii_path)
    except Exception as exc:
        logger.warning(f"状态图ASCII文本保存失败：{exc}")

    try:
        png_path = graph_dir / "kb_query_state_graph.png"
        png_path.write_bytes(graph.draw_mermaid_png())
        saved_paths["png"] = str(png_path)
    except Exception as exc:
        logger.warning(f"状态图PNG保存失败，仅保留Mermaid源文件：{exc}")

    logger.info(f"状态图已保存：{saved_paths}")
    return saved_paths


if __name__ == "__main__":
    save_query_graph()
