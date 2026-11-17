# Project Skills Overview

This repository contains a small AI assistant built with **LangChain**, **LangGraph**, and **Gradio**.  The core concepts demonstrated in the code are encapsulated as reusable *skills* that can be referenced by other Pi agents or by developers working on the project.

---

## 📚 Core Skills

| Skill | Location | Description |
|-------|----------|-------------|
| **Chat Interface** | `perplexia_ai/core/chat_interface.py` | Abstract base class defining `initialize()` and `process_message()` – the contract all week‑1 implementations follow. |
| **Week‑1 Factory** | `perplexia_ai/week1/factory.py` | Maps the three assignment modes (`part1`, `part2`, `part3`) to concrete chat classes. |
| **Query Understanding** | `perplexia_ai/week1/part1.py` | Uses a LangGraph `StateGraph` to classify user intent and format a response. |
| **Basic Tools (Calculator)** | `perplexia_ai/week1/part2.py` | Adds conditional routing to a calculator tool (`perplexia_ai/tools/calculator.py`). Also demonstrates Arize OpenTelemetry instrumentation. |
| **Conversation Memory (stub)** | `perplexia_ai/week1/part3.py` | Placeholder for a memory‑enabled chat implementation (to be completed). |
| **Calculator Tool** | `perplexia_ai/tools/calculator.py` | Safe evaluation of arithmetic expressions via a static `Calculator.evaluate_expression` method. |
| **Gradio Demo** | `perplexia_ai/app.py` | Builds a `gr.ChatInterface` that wires the selected week‑1 implementation into a web UI. |
| **CLI Entrypoint** | `run.py` | Simple command‑line wrapper that selects the part to run (`--mode part1|part2|part3`). |

---

## 🛠️ Supporting Skills

- **Environment handling** – `dotenv.load_dotenv()` loads API keys required for OpenAI, Gemini, Arize, etc.
- **Model initialization helper** – `langchain.chat_models.init_chat_model` is used in `part2.py` to obtain a Gemini chat model with a single call.
- **Observability** – `arize.otel.register` and `openinference.instrumentation.langchain.LangChainInstrumentor` provide tracing for the part‑2 workflow.

---

## 📦 How to Extend

1. **Add a new skill** – create a Python module under `perplexia_ai/` that implements `ChatInterface` and update `factory.py` with a new enum value.
2. **Document the skill** – add a row to the table above and, if the skill has non‑trivial usage, provide a short markdown section with example code.
3. **Expose via MCP** – if you want other Pi agents to discover the skill, place a `SKILL.md` file next to the implementation (as we have done for the existing skills in the `~/.agents/skills/` directory). The harness will automatically pick it up.

---

## 📖 References

- LangChain docs (MCP): `https://docs.langchain.com/mcp`
- LangGraph documentation: `https://langchain.com/langgraph`
- Gradio theming: `gr.themes.Soft()`

---

*This file is intended for both human readers and Pi’s skill‑discovery mechanism. Keep it up‑to‑date as the project evolves.*