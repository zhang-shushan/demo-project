from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def recall_with_scores(
        query: str,
        vectorstore: Chroma,
        k: int = 30
) -> Tuple[List[Document], List[float]]:
    """
    使用向量数据库进行检索，并返回文档和相似度分数
    :param query: 用户查询
    :param vectorstore: 向量数据库对象
    :param k: 返回的文档数量
    :return: 文档列表和相似度分数列表
    """
    # 检索文档及其相似度分数
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)

    docs = []
    scores = []
    for doc, score in results:
        doc.page_content = doc.page_content.strip()  # 去除首尾空格
        docs.append(doc)
        scores.append(score)

    return docs, scores
