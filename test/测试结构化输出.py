import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent_with_structured_output import IntentRecognizer

if __name__ == "__main__":
    # 不再用 langchain_community.chat_models.ChatTongyi
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="qwen3.7-flash",                       # 或 qwen-max-latest / qwen-plus
        api_key="sk-ws-H.ERLPLXM.YyYg.MEUCIQDq4Zlc5pSliXxAhoLkbxgg_bZRzR2AsZ_kUNF6yQuUBwIgOjM_ewt-ldS4vYQbNK0nSvOsa6Mw4zD3F6BHhAcF3RE",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

    recognizer = IntentRecognizer(llm)
    result = recognizer.recognize("我想查询我的订单，我的订单号是06715421bjfab412")
    print(result)  