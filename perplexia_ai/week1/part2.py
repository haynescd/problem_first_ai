"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from perplexia_ai.week1.common import (
    PromptAgentState,
    calculation,
    classify_intent,
    get_openai_llm,
    handle_query_prompts,
    route_after_classification,
)

tracer_provider = register(project_name="default", auto_instrument=True)


class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""

    def __init__(self):
        self.llm: ChatOpenAI | None = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.app: CompiledStateGraph | None = None
        self.calculator = Calculator()

    def initialize(self) -> None:
        """Initialize components for basic tools.

        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        self.llm = get_openai_llm()

        self._build_graph()

    def _build_graph(self) -> None:
        workflow = StateGraph(PromptAgentState)
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("calculation", calculation)
        workflow.add_node("handle_query_prompts", handle_query_prompts)

        workflow.add_edge(START, "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            route_after_classification,
            ["calculation", "handle_query_prompts"],
        )
        workflow.add_edge("calculation", END)
        workflow.add_edge("handle_query_prompts", END)

        self.app = workflow.compile()

    def process_message(
        self, message: str, chat_history: list[dict[str, str]] | None = None
    ) -> str:
        """Process a message with calculator support.

        Students should:
        - Check if calculation needed
        - Use calculator if needed
        - Otherwise, handle as regular query

        Args:
            message: The user's input message
            chat_history: Not used in Part 2

        Returns:
            str: The assistant's response
        """
        history = "\n".join(f"{m['role']}: {m['content']}" for m in (chat_history or []))

        runtime_config = {"configurable": {"model": self.llm}}
        input = PromptAgentState(user_input=message, history=history)

        final_state = self.app.invoke(input, config=runtime_config)
        return final_state.get("response")
