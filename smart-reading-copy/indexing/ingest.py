import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import QaConfig


class PdfIngestor:
    def __init__(self, config: QaConfig) -> None:
        self.config = config
    def ingest(self, pdf_bytes: bytes) -> None:
        # 二进制写入临时文件！
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(pdf_bytes)
            temp.flush()
            temp_path = temp.name
        try:
            # 加载 PDF
            documents = self.load(temp_path) # list[Document]
            # 拆分
            return self.split(documents)
        finally:
            os.unlink(temp_path)

    def load(self, pdf_path: str) -> List[Document]:
        """
        加载 PDF 文件，每页生成一个 Document 对象
        :param pdf_path: PDF 文件路径（由 storage.py 生成的临时文件）
        :return: Document 列表，每个 Document 包含 page_content 和 metadata（如页码、来源）
        """
        return PyMuPDFLoader(pdf_path).load()

    def split(self, docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.__config.chunk_size,
            chunk_overlap=self.__config.chunk_overlap,
            is_separator_regex=True,      # 启用正则分隔符
            separators=["(?<=。)", "(?<=！)", "(?<=？)", "(?<=；)", "(?<=，)", " ", "\n"],
        )
        return splitter.split_documents(docs)