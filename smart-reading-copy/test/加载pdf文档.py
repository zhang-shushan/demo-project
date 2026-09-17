from langchain_community.document_loaders import PyMuPDFLoader

pdf_path = 'C:\\PyCharm\\PyWorkSpace\\k-ai\\smart-reading-copy\\data\\sample_document.pdf'
pdf_loader = PyMuPDFLoader(pdf_path)
# 将pdf文档加载到内存中
pdf_documents = pdf_loader.load()

# <class 'list'>
print(type(pdf_documents))
# 总页数
print(len(pdf_documents))
# Document对象类型
print(type(pdf_documents[0]))

"""
page_content是Document类的一个属性，表示文档的内容。它是一个字符串类型，包含了文档的文本内容。
metadata是Document类的另一个属性，表示文档的元数据。它是一个字典类型，包含了文档的元信息。
"""
for doc in pdf_documents:
    print(f"page_content={repr(doc.page_content)}, metadata={doc.metadata}")
