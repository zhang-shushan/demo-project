import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat_service import ChatService
from core.protocol import ChatRequest
import uuid

# 创建服务实例
chat_service = ChatService()

# 固定 session_id 实现多轮对话
session_id = "test_session"
user_id = "test_user"

# 多轮对话列表
conversations = [
    "我的订单什么时候发货？",
    "ORD123456789",
    "能改地址吗？",
    "北京市朝阳区xxx",
    "谢谢",
    "没有了",
    "拜拜",
    "goodbye"
]

# 执行多轮对话
for round_num, user_input in enumerate(conversations, 1):
    request = ChatRequest(
        user_id=user_id,
        session_id=session_id,  # 使用相同的 session_id
        message_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        user_input=user_input
    )

    response = chat_service.handle(request)
    print(f"第{round_num}轮 - 用户: {user_input}")
    print(f"AI: {response}\n")