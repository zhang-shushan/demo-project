from pydantic import BaseModel, Field
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from .prompt import INTENT_RECOGNIZE_WITH_STRUCTURED_OUTPUT_PROMPT


class IntentResult(BaseModel):
    """意图识别结果：通过大语言模型识别用户输入的意图，输出结构化数据。"""
    intents: list[str] = Field(description="意图列表，每个元素为一个意图名称")
    slots: dict[str, Any] = Field(description="slot值字典，键为slot名称，值为slot值")
    confidence: float = Field(description="置信度分数，取值范围0~1，越大表示越确信该意图")

class IntentRecognizer:
    """
    意图识别器：通过大语言模型识别用户输入的意图，输出IntentResult对象
    """

    def __init__(self, llm) -> None:
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_RECOGNIZE_WITH_STRUCTURED_OUTPUT_PROMPT),
            ("human", "用户输入：{user_input}")
        ])
        self.__llm = llm
        self.__structured_llm = llm.with_structured_output(IntentResult)

    def recognize(self, user_input: str) -> IntentResult:
        chain = self.__prompt | self.__structured_llm
        result = chain.invoke(input={"user_input": user_input})

        # 如果返回 None，返回默认值
        if result is None:
            result = IntentResult(
                intents=["general"],
                slots={},
                confidence=0.0 
            )

        # 确保 confidence 在 0~1 范围内
        result.confidence = max(0.0, min(1.0, result.confidence))

        return result