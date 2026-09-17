from idlelib import history
from typing import Dict, List
from unittest import result

from dashscope.threads.messages import messages
from dotenv import get_key
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .prompt import SUMMARY_GENERATION_PROMPT

# 存储每个用户的对话历史：key是user_id:session_id，value是会话历史记录
_session_store: Dict[str, BaseChatMessageHistory] = {}
# 存储每个用户的会话摘要：key是user_id:session_id，value是会话摘要
_summary_store: Dict[str, str] = {}
# 存储每个用户的事实(slots)：key是user_id:session_id，value是事实
_facts_store: Dict[str, Dict[str, str]] = {}
# 存储每个用户的未摘要消息数量：key是user_id:session_id，value是未摘要消息数量
_unsummarized_count: Dict[str, int] = {}

# 每6轮（2*6=12）对话生成一次摘要、
SUMMARY_BATCH_SIZE = 12


# 记忆管理模块
class Memory:
    def __init__(self, user_id: str, session_id: str) -> None:
        self.__user_id = user_id
        self.__session_id = session_id
        self.__session_key = f"session:{user_id}:{session_id}"
        # 初始化摘要、事实、新消息的数量
        _summary_store[self.__session_key] = ""
        _facts_store[self.__session_key] = {}
        _unsummarized_count[self.__session_key] = 0

    def get_session_history(self) -> BaseChatMessageHistory:
        if self.__session_key not in _session_store:
            _session_store[self.__session_key] = InMemoryChatMessageHistory()
        return _session_store[self.__session_key]

    def __generate_incremental_summary(self,
                                       old_summary: str,
                                       new_messages: List[BaseMessage],
                                       llm=None) -> str:
        """
        生成增量会话摘要：将[旧摘要]和[新消息]合并后，调用大语言模型生成增量摘要
        :param old_summary: 旧的会话摘要
        :param new_messages: 新的会话历史记录
        :return: 增量会话摘要
        """
        # 如果没有新的消息，直接返回旧的摘要
        if not new_messages:
            return old_summary
        # 提取新消息的文本
        new_messages_text = ""
        for messsage in new_messages:
            if isinstance(messsage, HumanMessage):
                new_messages_text += f"用户: {messsage.content}\n"
            elif isinstance(messsage, AIMessage):
                new_messages_text += f"AI: {messsage.content}\n"
        # 如果没有LLM,那么直接使用简要的版本
        if not llm:
            return self._simple_incremental_summary(old_summary, new_messages_text)

        # 使用LLM生成增量摘要
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", SUMMARY_GENERATION_PROMPT),
            ("human", f"旧摘要：{old_summary}\n新消息：{new_messages_text}")
        ])
        try:
            chain = summary_prompt | llm | StrOutputParser()
            result = chain.invoke(input={})
            new_summary = result.strip()
            return new_summary
        except:
            return self._simple_incremental_summary(old_summary, new_messages_text)

    def _simple_incremental_summary(self,
                                    old_summary: str,
                                    new_messages_text: str) -> str:
        """
        简单的增量会话摘要：将[旧摘要]和[新消息]合并后，返回合并后的摘要
        :param old_summary: 旧的会话摘要
        :param new_messages_text: 新的会话历史记录
        :return: 增量会话摘要
        """
        if old_summary and old_summary != "[摘要] 无":
            # 截取旧摘要的前150字 + 新消息的前150字
            old_part = old_summary[:150] if len(old_summary) >= 150 else old_summary
            new_part = new_messages_text[:150] if len(new_messages_text) >= 150 else new_messages_text
            return f"[摘要] {old_part}...{new_part}"
        else:
            return f"[摘要] {new_messages_text[:150]}"

    def _trim_and_summarize(self, llm=None):
        history = self.get_session_history()
        if not history:
            return

        unsummarized = _unsummarized_count.get(self.__session_key, 0)
        # 新消息达到阈值，就触发摘要更新
        if unsummarized >= SUMMARY_BATCH_SIZE:
            messages = history.messages
            total_count = len(messages)
            summarized_count = total_count - unsummarized

            # 获取未总结的消息
            new_messages = messages[summarized_count:]
            if new_messages:
                old_summary = _summary_store.get(self.__session_key,"")
                # 增量生成摘要信息
                new_summary = self.__generate_incremental_summary(
                    old_summary,
                    new_messages,
                    llm
                )
                # 更新摘要
                _summary_store[self.__session_key] = new_summary
                # 重置未摘要的消息数量
                _unsummarized_count[self.__session_key] = 0
                print(f"摘要生成：{new_summary}")

                # 删除旧消息
    def add_user_message(self, message: str, llm=None):
        """
        添加用户消息到会话历史，并更新摘要
        :param message: 用户消息
        :param llm: 大语言模型
        :return:
        """

        history = self.get_session_history()
        history.add_user_message(message)

        # 增加未摘要消息的数量（新消息数量）
        _unsummarized_count[self.__session_key] += 1
        # 触发摘要更新
        self._trim_and_summarize(llm)

    def add_ai_message(self, message: str, llm=None) -> None:
        """
        添加AI消息到会话历史记录
        :param message: AI消息
        """
        history = self.get_session_history()
        history.add_ai_message(message)

        # 增加未摘要消息的数量
        _unsummarized_count[self.__session_key] += 1
        # 触发摘要更新
        self._trim_and_summarize(llm)

    def get_key_facts(self) -> Dict[str, str]:
        """
        获取会话的关键事实(slots)
        :return: 关键事实字典
        """
        return _facts_store.get(self.__session_key, {}).copy()

    def update_key_facts(self, new_facts: Dict[str, str]):
        """
        更新会话的关键事实(slots)
        :param new_facts: 新的关键事实字典
        """
        if self.__session_key not in _facts_store:
            _facts_store[self.__session_key] = {}

        # 更新关键事实
        _facts_store[self.__session_key].update({
            k: v for k, v in new_facts.items() if v
        })
        """
        _facts_store: Dict[str, Dict[str, str]] = {}
        _facts_store: {user1:session1 : {order_id:1}}
        
        _facts_store[user1:session1].update({order_id:2})
        _facts_store: {user1:session1 : {order_id:2}}
        """

    def clear_key_facts(self) -> None:
        """
        清除当前会话的关键事实
        :return:
        """
        if self.__session_key in _facts_store:
            del _facts_store[self.__session_key]

    def prepare_memory_for_llm(self) -> List[BaseMessage]:
        history = self.get_session_history()
        messages = history.messages
        if len(messages) == 14:
            print("记忆准备")

        result = []
        # 获取当前会话的摘要和关键事实,近期会话
        # 1.添加关键事实
        facts = self.get_key_facts()
        if facts:
            facts_text = " | ".join([f"{key}:{value}" for key,value in facts.items()])
            result.append(AIMessage(content=f"[关键事实：]{facts_text}"))
        # 2.添加摘要
        summary = _summary_store.get(self.__session_key, "")
        if summary:
            result.append(AIMessage(content=f"[摘要：]{summary}"))
        # 3.添加近期会话
        unsummarized_count = _unsummarized_count.get(self.__session_key, 0)
        if unsummarized_count > 0:
            resent_messages = messages[-unsummarized_count:]
            result.extend(resent_messages)
        return result









