from modelscope.hub.snapshot_download import snapshot_download

local_dir = r"\home\wang\agent\dataset_rag"

snapshot_download(
    model_id="BAAI/bge-reranker-large",
    cache_dir=local_dir,
)

print("下载完成，模型目录：", local_dir)