from enum import StrEnum

from langchain.chat_models import init_chat_model
from langchain_core.messages import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph.state import RunnableConfig
from pydantic import BaseModel, Field

from perplexia_ai.tools import calculator


class Classification(StrEnum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    CALCULATION = "calculation"
    ANALYTICAL = "analytical"
    DEFINITION = "definition"


class IntentClassification(BaseModel):
    intent: Classification = Field(
        description="The matching classification category that best captures the user's core request."
    )


class PromptAgentState(BaseModel):
    user_input: str | None = None
    classification: Classification | None = None
    prompt: str | None = None
    response: str | None = None
    history: str


CLASSIFICATION_PROMPTS = {
    Classification.FACTUAL: "If the user starts the question with 'What is or Who invented'",
    Classification.COMPARISON: "If the user starts the question with 'What is the difference between'",
    Classification.ANALYTICAL: "If the user starts the question with 'How does or Why do'",
    Classification.DEFINITION: "If the user starts the question with 'Define or Explain'",
    Classification.CALCULATION: "If the user is asking to solve a math problem, calculate a number, compute percentages, tips, or perform any arithmetic operation (e.g., 'Compute 17% tip', 'What is 45 * 12').",
}


def build_system_prompt() -> str:
    prompt = "You are an intent classifier.Return ONLY the category name. Classify the user query into one these categories:"
    for category, p in CLASSIFICATION_PROMPTS.items():
        prompt += f"\n - {category}: {p}"
    return prompt


def classify_intent(state: PromptAgentState, config: RunnableConfig):
    query_classifier_prompt = ChatPromptTemplate.from_messages(
        [("system", f"{build_system_prompt()}"), ("human", "{user_input}")]
    )

    llm = config.get("configurable", {}).get("model")
    if not llm:
        raise ValueError("LLM must be provided via graph configuration.")
    structured_llm = llm.with_structured_output(IntentClassification)
    classifier_chain = query_classifier_prompt | structured_llm
    output = classifier_chain.invoke({"user_input": state.user_input})

    print(output.intent)
    state.classification = output.intent
    return state.model_dump()


def handle_query_prompts(state: PromptAgentState, config: RunnableConfig):
    prompt_map = {
        Classification.FACTUAL: "You are answering factual type questions. Your answers should be concise and direct",
        Classification.COMPARISON: "You are answering comparison type questions. Your answers should use structured formats (tables, bullet points)",
        Classification.ANALYTICAL: "You are answering analytical type questions. Your answers should include reasoning steps",
        Classification.DEFINITION: "You are answering definition type questions. Your answers (Define..., Explain...)",
    }

    history_prompt = """Use information from the conversation history only if relevant to the above user query, otherwise ignore the history.
        Conversation history with the user:
        {history}"""

    p = prompt_map[state.classification]
    prompt = ChatPromptTemplate.from_messages(
        [("system", f"{p}\n{history_prompt}"), ("human", "{user_input}")]
    )

    llm = config.get("configurable", {}).get("model")
    if not llm:
        raise ValueError("LLM must be provided via graph configuration.")

    chain = prompt | llm | StrOutputParser()
    output = chain.invoke({"user_input": state.user_input, "history": state.history})

    state.response = output
    return state.model_dump()


def calculation(state: PromptAgentState, config: RunnableConfig):
    llm = config.get("configurable", {}).get("model")
    if not llm:
        raise ValueError("LLM must be provided via graph configuration.")

    system_prompt = """
    Extract ONLY the mathematical expression.
    the extracted mathematical expression must only contain allowed characters (digits, spaces, +, -, *, /, (, ), and .).
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", f"{system_prompt}"), ("human", "{user_input}")]
    )
    chain = prompt | llm | StrOutputParser()
    output = chain.invoke({"user_input": state.user_input})

    state.response = str(calculator_tool(output))
    return state.model_dump()


def calculator_tool(expression: str):
    return calculator.Calculator.evaluate_expression(expression=expression)


def route_after_classification(state: PromptAgentState):
    if state.classification == Classification.CALCULATION.value:
        return "calculation"
    else:
        return "handle_query_prompts"
