"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from phoenix.otel import register

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.week1.common import (
    PromptAgentState,
    classify_intent,
    get_openai_llm,
    handle_query_prompts,
)

tracer_provider = register(project_name="default", auto_instrument=True)


class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm: ChatOpenAI | None = None
        self.app: CompiledStateGraph | None = None

    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """

        self.llm = get_openai_llm()
        self._build_graph()

    def _build_graph(self) -> None:
        workflow = StateGraph(PromptAgentState)
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("handle_query_prompts", handle_query_prompts)

        workflow.add_edge(START, "classify_intent")
        workflow.add_edge("classify_intent", "handle_query_prompts")
        workflow.add_edge("handle_query_prompts", END)
        self.app = workflow.compile()
        print(self.app.get_graph().draw_ascii())

    def process_message(
        self, message: str, chat_history: list[dict[str, str]] | None = None
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
        input = PromptAgentState(user_input=message, history="")

        final_state = self.app.invoke(input, config=runtime_config)
        return final_state.get("response")
