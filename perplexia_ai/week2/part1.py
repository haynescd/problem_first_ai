"""
Part 1 - Web Search implementation using LangGraph with Tracing.

This implementation focuses on:
- Setting up web search using Tavily
- Processing search results
- Formatting responses with citations
"""

from pprint import pformat
from typing import Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from perplexia_ai.core.chat_interface import ChatInterface


class SearchResultItem(BaseModel):
    title: str = Field(description="The short, catchy headline for the breakthrough or milestone.")
    description: str = Field(
        description="A concise, one-sentence summary explaining the development."
    )


class SourceItem(BaseModel):
    source_name: str = Field(
        description="The publisher or platform name (e.g., MIT Technology Review)."
    )
    date: str = Field(description="The month and year of the publication (e.g., April 2023).")
    link: str = Field(description="The URL provided in the search context.")


class SearchSummary(BaseModel):
    developments: List[SearchResultItem] = Field(
        description="A list of the latest developments found in the search results."
    )
    sources: List[SourceItem] = Field(
        description="A list of all unique sources utilized to back up the facts."
    )


class WebSearchState(BaseModel):
    query: str
    results: dict | None = None
    tavily_answer: str | None = None
    summary: SearchSummary | None = None


summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert research analyst. Your task is to analyze raw search results "
                "and extract the key developments. \n\n"
                "CRITICAL RULES:\n"
                "- Only extract facts directly supported by the search results.\n"
                "- Do not make up dates or links; if a link is missing, use an empty string or the base domain.\n"
                "- Synthesize the information into a few distinct, impactful items."
            ),
        ),
        ("human", "User Question: {question}\n\nRaw Search Results:\n{search_results}"),
    ]
)


class WebSearchChat(ChatInterface):
    """Perplexia Journey - Week 2 Part 1 implementation for web search using LangGraph."""

    def __init__(self):
        self.llm = None
        self.search_tool = None
        self.graph = None

    def initialize(self) -> None:
        """Initialize components for web search and tracing.

        Students should:
        - Initialize the LLM
        - Set up Tavily search tool
        - Create a LangGraph workflow for web search
        """

        self.llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")

        self.search_tool = TavilySearch(
            max_results=3,
            include_answer=False,
            include_raw_content=True,
            search_depth="advanced",
            include_images=False,
        )

        def web_search(state: WebSearchState):
            results = self.search_tool.invoke({"query": state.query})
            state.results = results
            return state.model_dump()

        def llm_generate_summary(state: WebSearchState):
            structured_llm = self.llm.with_structured_output(SearchSummary)
            chain = summary_prompt | structured_llm

            output = chain.invoke(
                {"question": state.query, "search_results": pformat(state.results)}
            )
            state.summary = output
            return state.model_dump()

        graph_builder = StateGraph(WebSearchState)
        graph_builder.add_node("web_search", web_search)
        graph_builder.add_node("llm_generate_summary", llm_generate_summary)

        graph_builder.add_edge(START, "web_search")
        graph_builder.add_edge("web_search", "llm_generate_summary")
        graph_builder.add_edge("llm_generate_summary", END)

        self.graph = graph_builder.compile()

    @staticmethod
    def _format_summary(summary: dict) -> str:
        print(pformat(summary))
        lines = []
        for i, dev in enumerate(summary["developments"], 1):
            lines.append(f"{i}. {dev['title']}")
            lines.append(f"   {dev['description']}")
            lines.append("")
        lines.append("Sources:")
        for src in summary["sources"]:
            lines.append(f"- {src['source_name']} ({src['date']}) ({src['link']})")
        return "\n".join(lines).strip()

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Process a message using web search and record trace.

        Args:
            message: The user's input message
            chat_history: Previous conversation history

        Returns:
            str: The assistant's response based on web search results
        """
        input = WebSearchState(query=message)
        final = self.graph.invoke(input)
        summary = final.get("summary")
        return self._format_summary(summary)
