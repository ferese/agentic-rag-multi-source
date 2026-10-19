Your current project: decides which source to use, retrieves information, evaluates relevance, and can retry when the evidence is inadequate.
==================================================================================================================================================

Agent architecture
LangGraph graph with routing, retrieval, grading, generation and correction.

RAG implementation
Document ingestion, embeddings, Chroma vector search and evidence.

Tool integration
Vector retrieval, SQLite queries and DuckDuckGo search.

Self-correction
Retrieval grading, query rewriting, grounding checks and bounded retries.
===================================================================================

Your pyproject.toml confirms that the project uses LangGraph, LangChain, Ollama, ChromaDB and local Hugging Face embeddings. We don't need to add a paid API or another framework.

===================================================================================

### Proposed Agent Architecture

The proposed system is a local, multi-source Agentic Retrieval-Augmented Generation (RAG) application implemented using LangGraph and Streamlit. It accepts a user query and determines whether a single-source workflow or a multi-step execution plan is required.

The system provides three tools: an SQLite database tool for structured queries, a Chroma vectorstore for retrieving information from indexed documents, and a web-search tool for accessing public web information. In multi-step workflows, outputs from earlier steps can be passed to subsequent tools as context.

The collected evidence is provided to a locally hosted language model through Ollama to generate the final response. The application displays the answer, execution plan, tool status, and supporting evidence. The design aims to improve transparency and reduce unsupported responses by using retrieved information and reporting when relevant evidence is unavailable.

================================================================================================================================================================================================

A conventional chatbot does not inherently query structured databases, search a private document collection, or show how it reached an answer. Users need a system that can route questions to appropriate sources, combine structured and unstructured evidence, handle missing information honestly, and expose its steps for inspection. The project addresses this need through a local agentic workflow that integrates SQL, vector retrieval and web search.
