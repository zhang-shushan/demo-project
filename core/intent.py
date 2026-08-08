import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .prompt import INTENT_RECOGNIZE_PROMPT


@dataclass(frozen=True)
class IntentResult:
    """
    意图识别结果：
    - intents: 意图列表, 每个元素为一个意图名称
    - slots: slot值字典, 键为slot名称, 值为slot值
    - confidence: 置信度分数, 取值范围 0~1, 越大表示越信该意图l
    """
    intents: list[str]
    slots: dict[str, Any]
    confidence: float


class IntentRecognizer:
    """
    意图识别器：通过大语言模型识别用户输入的意图，输出IntentResult对象
    """

    def __init__(self, llm):
        """
        初始化意图识别
        :param llm: 大语言模型
        """
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system", INTENT_RECOGNIZE_PROMPT),
            ("human", "用户输入: {user_input}")
        ])
        self.__llm = llm
        self.__str_output_parser = StrOutputParser()

    def recognize(self, user_input: str) -> IntentResult:
        """
        识别用户输入的意图
        :param user_input: 用户输入的文本
        :return: 意图识别结果
        """
        # 1.调用llm去识别用户的意图，输出str格式的json字符串
        chain = self.__prompt | self.__llm | self.__str_output_parser
        result = chain.invoke(input={"user_input": user_input})
        # 2.解析llm的输出
        data = self.__parse_str_to_json(result)
        # 2.1.解析intents意图
        intents = data.get("intents")
        if not isinstance(intents, list):
            intents = [intents] if isinstance(intents, str) else ["general"]
        # 2.2.解析slots插槽数据
        slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
        # 2.3.解析confidence置信度
        confidence = data.get("confidence")
        try:
            confidence = float(confidence)
        except ValueError:
            confidence = 0.0
        confidence = max(0.0,min(1.0, confidence))

        return IntentResult(intents, slots, confidence)

    def __parse_str_to_json(self, text: str) -> dict[str, Any]:
        if not text and not text.strip():
            return {"intent": ["general"], "slots": {}, "confidence": 0.0}

        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecoder:
            pass

        # 如果解析失败，尝试从模型输出中提取json字符串
        find_text = re.search(r"{{.*?}}", text, re.DOTALL)
        if find_text:
            try:
                return json.loads(find_text.group(0))
            except json.JSONDecodeError:
                pass
        return {"intent": ["general"], "slots": {}, "confidence": 0.0}
