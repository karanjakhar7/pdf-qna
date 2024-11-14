
# PDF Question Answering System

  

This project provides a system for asking questions about PDF documents using LLMs. It includes both a FastAPI backend service, a Streamlit web interface, and a python script that can be run in the terminal.

  

## Features

- Upload PDF documents

- Ask questions about the content of PDFs

- Get AI-generated answers based on the document content

- Support for both single questions and batch questions
  

## Components

### RAG Setup (`core/`)
- OpanAI text-embedding-3-small for embeddings
- FAISS as in-memory vector DB
- OpenAI gpt-4o-mini as generator


### FastAPI Backend (`app/api.py`)
- REST API endpoint for PDF processing and question answering
- Handles batch questions
- Automatic cleanup of temporary files
- Input validation using Pydantic models


### Streamlit Frontend (`app/streamlit.py`)
- User-friendly web interface
- PDF file upload functionality
- Interactive question-answer interface
- Session state management for persistent PDF processing

  

## Setup
1.  The project uses Azure OpenAI models. If you need to use OpenAI models directly, please update the provider in the  `config.py`  file. Create `.env` file in `app/` dir.
	- To use AzureOpenAI, keep the following environment variables:    
	    - AZURE_OPENAI_ENDPOINT
	    -   AZURE_OPENAI_API_KEY  
     - To use OpenAI models, keep the following environment variables:    
	    - OPENAI_API_KEY
	    - 
2.  Install uv if not already installed.  [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
    
3.  Install all the dependencies:  `uv sync`
    
4.  Start the servers:
	-	 FastAPI server:  `uv run fastapi run`
	-	Streamlit application: `uv run streamlit run app/streamlit.py`
	-	Terminal script: `uv run python app/run_terminal.py`


### FASTAPI Endpoint:
Send a POST request to `/generate_answers` with:
- PDF file
- List of questions