"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from enum import StrEnum
from gettext import Catalog
from typing import Dict, List, Optional, TypedDict
from unicodedata import category

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.week1.common import (
    Classification,
    PromptAgentState,
    classify_intent,
    handle_query_prompts,
)


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""

    def __init__(self):
        self.llm = None
        self.app = None

    def initialize(self) -> None:
        """Initialize components for query understanding.

        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """

        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
        workflow = StateGraph(PromptAgentState)
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("handle_query_prompts", handle_query_prompts)

        workflow.add_edge(START, "classify_intent")
        workflow.add_edge("classify_intent", "handle_query_prompts")
        workflow.add_edge("handle_query_prompts", END)

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using query understanding.

        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type

        Args:
            message: The user's input message
            chat_history: Not used in Part 1

        Returns:
            str: The assistant's response
        """
        runtime_config = {"configurable": {"model": self.llm}}
        input = PromptAgentState(user_input=message)

        final_state = self.app.invoke(input, config=runtime_config)
        return final_state.get("response")
