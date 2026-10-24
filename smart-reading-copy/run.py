# run.py - PDF 问答系统（支持逐条消息，自动维护对话历史）
import os
from typing import Dict, Any, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline
from querying.rag_pipeline import RagPipeline


class PdfQA:
    def __init__(self, pdf_bytes: bytes, api_key: Optional[str] = None):
        self.pdf_bytes = pdf_bytes
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY enviroment variable not set")
        self.api_key = api_key

        self.config = QaConfig()
        self.file_hash: Optional[str] = None
        self.chat_history: Optional[InMemoryChatMessageHistory] = None
        self.rag: Optional[RagPipeline] = None

    def ask(self, question: str):
        # 建立知识库
        if self.file_hash is None:
            self._build_index()

        result = self.rag.query(
            dashscope_api_key=self.api_key,
            file_hash=self.file_hash,
            question=question,
            chat_history=self.chat_history.messages,
            enable_rewrite=True,
            enable_rerank=True,
            recall_k=30,
            hybrid_top_m=12,
            top_k=4
        )
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(result['answer'])
        return result

    def _build_index(self) -> None:
        if self.file_hash is not None:
            return  # 已经构建过
        print("正在构建索引（首次运行会调用 Embedding API，请稍候）...")
        indexer = IndexingPipeline(self.config, self.api_key)
        self.file_hash = indexer.build_from_file(self.pdf_bytes, self.api_key)
        self.chat_history = InMemoryChatMessageHistory()  # List[BaseMessage]
        self.rag = RagPipeline(self.config)
        print(f"索引构建完成，file_hash: {self.file_hash}")
