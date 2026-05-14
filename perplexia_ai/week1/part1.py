"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from enum import StrEnum
from gettext import Catalog
from typing import Dict, List, Optional
from unicodedata import category

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from perplexia_ai.core.chat_interface import ChatInterface


class Classification(StrEnum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    CALCULATION = "calculation"
    ANALYTICAL = "analytical"
    DEFINITION = "definition"


CLASSIFICATION_PROMPTS = {
    Classification.FACTUAL: "If the user starts the question with 'What is or Who invented'",
    Classification.COMPARISON: "If the user starts the question with 'What is the difference between'",
    Classification.ANALYTICAL: "If the user starts the question with 'How does or Why do'",
    Classification.DEFINITION: "If the user starts the question with 'Define or Explain'",
}


def build_system_prompt() -> str:
    prompt = "You are an intent classifier.Return ONLY the category name. Classify the user query into one these categories:"
    for category, p in CLASSIFICATION_PROMPTS.items():
        prompt += f"\n - {category}: {p}"
    return prompt


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""

    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}

    def initialize(self) -> None:
        """Initialize components for query understanding.

        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """

        self.query_classifier_prompt = ChatPromptTemplate.from_messages(
            [("system", f"{build_system_prompt()}"), ("human", "{user_input}")]
        )
        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

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
        classifier_chain = self.query_classifier_prompt | self.llm | StrOutputParser()
        category = classifier_chain.invoke({"user_input": message})

        return category
