# SAP Service Layer OData Query Generator

A Python application that generates OData queries for SAP Business One Service Layer from natural language questions using AI and RAG (Retrieval-Augmented Generation). Purpose of this project is to help users to generate OData queries for SAP Business One Service Layer from natural language questions using AI and RAG (Retrieval-Augmented Generation) And pass this to Model context protocol(MCP) to generate data based on OData query.

It is sample, can be used as a starting point for building a more complex application. AI Agent which can search data bases on the prompt. For Example,

1. User ask "Find Business Partner with name 'John Doe'"
2. AI Agent will search data from SAP documentation and generate OData query for Business Partner with name 'John Doe'
3. AI Agent will pass this OData query to Model context protocol(MCP) to generate data based on OData query
4. AI Agent will return the data to the user

## Features

- **Natural Language to OData**: Convert plain English questions into valid OData queries for SAP Service Layer
- **Hybrid Search**: Uses BM25 + FAISS for accurate context retrieval from SAP documentation
- **Web Interface**: User-friendly Gradio interface for easy interaction
- **Date-Aware Queries**: Automatically handles relative date expressions like "this month" or "last week"

## Requirements

- Python 3.12+
- Anthropic API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd service-layer-query-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
uv pip install -e .
```

4. Create a `.env` file with your configuration:
```env
ANTHROPIC_API_KEY=your-api-key-here
ANTHROPIC_MODEL_ID=claude-3-5-haiku-20241022
TEMPERATURE=0.7
MAX_NEW_TOKENS=1024
```

## Usage

### Run the Web App

```bash
python main.py
```

The Gradio interface will launch at `http://localhost:7861`.

### Example Questions

- "Find Business Partner with name 'John Doe'"
- "Get all open sales orders from customer C20000"
- "List top 10 items with price greater than 100"
- "Find all invoices from July 2025"

### CLI Usage

```bash
python src/test_odata_servicelayer.py
```

## Project Structure

```
service-layer-query-generator/
├── main.py                         # Entry point for the web app
├── SAP_SL_Rule_Book_2.txt          # SAP Service Layer documentation/rules
├── src/
│   ├── persentation.py             # Gradio UI
│   ├── test_odata_servicelayer.py  # OData query generation logic
│   └── modules/
│       ├── data_extract.py         # Text data extraction
│       ├── llm_model.py            # Anthropic LLM setup
│       ├── embedding_model.py      # HuggingFace embeddings
│       ├── langchain_data.py       # Text chunking
│       ├── prompts.py              # Prompt templates for OData generation
│       ├── retriever.py            # FAISS retrieval logic
│       └── hybrid_retriever.py     # BM25 + FAISS hybrid search
├── pyproject.toml
└── .env
```

## Tech Stack

- **LLM**: Anthropic Claude (via LangChain)
- **Embeddings**: HuggingFace sentence-transformers
- **Vector Store**: FAISS
- **Keyword Search**: BM25 (rank-bm25)
- **UI**: Gradio

# Ask for help

If you have any questions or need help, feel free to ask in the [GitHub issues](https://github.com/your-username/service-layer-query-generator/issues).
I will share the RAG Service Layer Query Generator rules book with you. 

## License

MIT License