# run.py - PDF 问答系统（支持逐条消息，自动维护对话历史）
import os
from typing import Dict, Any, Optional

from langchain_core.chat_history import InMemoryChatMessageHistory
from config.setting import QaConfig
from indexing.indexing_pipeline import IndexingPipeline
from querying.rag_pipeline import RagPipeline


class PDFQA:
    """
    PDF 问答系统封装类，支持逐条消息交互，自动维护对话历史
    """

    def __init__(self, pdf_bytes: bytes, api_key: Optional[str] = None):
        """
        构造函数
        :param pdf_bytes: PDF 文件的二进制数据
        :param api_key: 通义千问 API Key，若为 None 则从环境变量 DASHSCOPE_API_KEY 读取
        """
        self.pdf_bytes = pdf_bytes
        # 处理 API Key
        if api_key is None:
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not api_key:
                raise ValueError("未提供 API Key，且环境变量 DASHSCOPE_API_KEY 未设置")
        self.api_key = api_key

        self.cfg = QaConfig()
        self.file_hash: Optional[str] = None
        self.chat_history: Optional[InMemoryChatMessageHistory] = None
        self.rag: Optional[RagPipeline] = None

    def _build_index(self) -> None:
        """构建向量索引（内部懒加载）"""
        if self.file_hash is not None:
            return  # 已经构建过
        print("正在构建索引（首次运行会调用 Embedding API，请稍候）...")
        indexer = IndexingPipeline(self.cfg)
        self.file_hash = indexer.build_from_bytes(self.pdf_bytes, self.api_key)
        self.chat_history = InMemoryChatMessageHistory()
        self.rag = RagPipeline(self.cfg)
        print(f"索引构建完成，file_hash: {self.file_hash}")

    def ask(self, question: str) -> Dict[str, Any]:
        """
        发送一条问题，返回答案（自动维护对话历史）
        :param question: 用户问题
        :return: 包含 answer, context, evidence, search_query, rewritten 的字典
        """
        # 懒加载索引
        if self.file_hash is None:
            self._build_index()

        result = self.rag.query(
            dashscope_api_key=self.api_key,
            file_hash=self.file_hash,
            question=question,
            chat_history=self.chat_history.messages,
            enable_rewrite=True,
            enable_rerank=True,
            top_k=4,
            recall_k=30,
            hybrid_top_m=12,
        )

        # 更新对话历史
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(result['answer'])

        return result
