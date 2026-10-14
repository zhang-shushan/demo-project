# indexing/pipeline.py
"""
索引构建流水线：PDF → 切分 → Embedding → 持久化 Chroma
输出：file_hash（用于后续查询）
"""
from langchain_community.embeddings import DashScopeEmbeddings

from config.setting import QaConfig
from indexing.ingest import PdfIngestor
from indexing.storage import PdfBytesStore
from indexing.vectorstore import VectorStoreManager


class IndexingPipeline:
    def __init__(self,config: QaConfig) -> None:
        self.config = config
        self.store = PdfBytesStore()
        self.ingestor = PdfIngestor(config)

    def build_from_bytes(self, pdf_bytes: bytes, dashscope_api_key: str) -> str:
        # 1. 计算 hash，并且存储临时文件（将hash值作为临时文件的名字）
        file_hash = self.store.compute_hash(pdf_bytes)
        self.store.store(pdf_bytes, file_hash)
        # 2. 加载&切分文档
        chunks = self.ingestor.ingest(pdf_bytes)
        # 3. 构建/加载向量库（传入 chunks 会新建）
        embeddings = DashScopeEmbeddings(
            model=self.config,
            dashscope_api_key=dashscope_api_key
        )
        vector_store = VectorStoreManager(self.config, embeddings)
        vector_store.load_or_build(file_hash, chunks)
        return file_hash

    def build_from_file(self, pdf_path: str, dashscope_api_key: str) -> str:
        """从文件路径构建索引"""
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        return self.build_from_bytes(pdf_bytes, dashscope_api_key)
