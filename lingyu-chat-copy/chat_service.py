import logging
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from config import *
from core.intent import IntentRecognizer
from core.memory import Memory
from core.protocol import ChatRequest, ChatResponse

log = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.3


def _system_prompt_by_intent(intent: str) -> str:
    mapping = {
        "track_shipping": "你是电商物流查询客服。先要订单号/运单号；如果缺失就追问。",
        "change_address": "你是电商改地址客服。先确认是否已发货；需要订单号+新地址。",
        "refund": "你是电商退货退款客服。先给3步结论，再给注意事项，最后引导用户提供订单号。",
        "complaint": "你是电商投诉处理客服。先安抚，再收集订单号与问题细节，给出处理时效。",
        "general": "你是一个友好、简洁的聊天助手。",
    }
    return mapping.get(intent.strip(), mapping["general"])


def _handle_low_confidence(intent_result, user_input: str) -> str:
    return "抱歉，我没太理解您的需求。请问您是想：\n" \
           "1️⃣ 查询物流进度\n" \
           "2️⃣ 修改收货地址\n" \
           "3️⃣ 申请退货退款\n" \
           "4️⃣ 投诉建议\n" \
           "请回复数字或具体需求。"


class ChatService:
    def __init__(self, api_key: str = None) -> None:
        # 初始化日志记录器
        if not logging.root.handlers:
            logging.basicConfig(level=logging.INFO)
        # 初始化llm,需要用户传递api_key,否则使用 config 中的 TONGYI_API_KEY
        resolved_key = api_key or TONGYI_API_KEY
        self.__llm = ChatOpenAI(
            model=TONGYI_MODEL,
            api_key=resolved_key,
            base_url=TONGYI_BASE_URL,
            temperature=TONGYI_TEMPERATURE,
            max_tokens=TONGYI_MAX_OUTPUT_LENGTH,
        )
        self.__intent_recognizer = IntentRecognizer(self.__llm)
        # 维护 Memory 对象缓存：key是user_id:session_id，value是Memory对象
        self.__memory_cache: dict[str, Memory] = {}

    def __get_memory(self, user_id: str, session_id: str) -> Memory:
        key = f"{user_id}:{session_id}"
        if key not in self.__memory_cache:
            self.__memory_cache[key] = Memory(user_id, session_id)
        return self.__memory_cache[key]

    def handle(self, request: ChatRequest) -> ChatResponse:
        # 请求开始时间
        start_time = time.time()

        try:
            # 1.意图识别
            intent_result = self.__intent_recognizer.recognize(request.user_input)
            # 2.低置信度处理
            if intent_result.confidence < CONFIDENCE_THRESHOLD:
                low_conf = _handle_low_confidence(intent_result, request.user_input)
                # 低置信度回合也要写入记忆，否则该轮不计入摘要的触发计数，
                # 导致"每6轮生成一次摘要"的时机被低置信度轮次打乱
                memory = self.__get_memory(request.user_id, request.session_id)
                memory.add_user_message(request.user_input, self.__llm)
                memory.add_ai_message(low_conf, self.__llm)
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=request.session_id,
                    message_id=request.message_id,
                    trace_id=request.trace_id,
                    response=low_conf,
                )
            # 3.确定主要意图
            intents = intent_result.intents
            primary_intent = intents[0] if intents else "general"
            # 4.构建prompt(不同的意图选择不同的prompt，也就是选择不同的链路)
            prompt = ChatPromptTemplate.from_messages([
                ("system", _system_prompt_by_intent(primary_intent)),
                MessagesPlaceholder("history"),
                ("human", "用户输入：{user_input}")
            ])
            # 5.记忆管理
            memory = self.__get_memory(request.user_id, request.session_id)

            def prepare(input_dict: dict) -> dict:
                # 准备提供给大语言模型的上下文
                memory_messages = memory.prepare_memory_for_llm()
                return {
                    "user_input": input_dict["user_input"],
                    "history": memory_messages
                }

            # 6.LLM调用(记忆准备->Memory)
            chain = RunnableLambda(prepare) | prompt | self.__llm | StrOutputParser()
            answer = chain.invoke(input={"user_input": request.user_input})
            # 7. 写回历史消息（user消息 和 AI消息）
            memory.add_user_message(request.user_input, self.__llm)
            memory.add_ai_message(answer, self.__llm)
            # 8.更新事实
            memory.update_key_facts(intent_result.slots)
            # 9.日志记录
            self.__log(
                req=request,
                intents=intents,
                confidence=intent_result.confidence,
                action="normal",
                latency_ms=int((time.time() - start_time) * 1000),
            )

            return ChatResponse(
                user_id=request.user_id,
                session_id=request.session_id,
                message_id=request.message_id,
                trace_id=request.trace_id,
                response=answer
            )

        except Exception as e:
            # 异常处理，记录错误日志
            self.__log(
                req=request,
                intents=["error"],
                confidence=0.0,
                action="error",
                latency_ms=int((time.time() - start_time) * 1000),
                error=str(e),
            )
            raise e

    def __log(self,
              *,
              req: ChatRequest,
              intents: list[str],
              confidence: float,
              action: str,
              latency_ms: int,
              error: str | None = None):
        payload = {
            "trace_id": req.trace_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
            "message_id": req.message_id,
            "intents": intents,
            "confidence": confidence,
            "action": action,
            "latency_ms": latency_ms
        }
        # 如果有错误信息，记录为错误日志error，否则记录为普通日志info
        if error:
            payload["error"] = error
            log.error(payload)
        else:
            log.info(payload)
