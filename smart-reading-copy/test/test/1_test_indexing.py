from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline

config = QaConfig()
pipeline = IndexingPipeline(config)

pdf_path = "你的pdf文件位置。"
file_hash = pipeline.build_from_file(pdf_path, dashscope_api_key="你的dashscope_api_key")
print(f"索引构建完成，file_hash: {file_hash}")