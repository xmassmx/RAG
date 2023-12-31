# Retrieval Augmented Generation (RAG)

This repository contains the source code and resources for a Retrieval Augmented Generation (RAG) application. The application is a basic Q/A bot that answers questions about some of my projects and assignments from my undergraduate studies. The aim of this project is to learn how to build an LLM application, more specifically a Retrieval Augmented Generation system and understand and optimise its constituent modules.
The codebase for the LLM Apps course from Weights and Biases is used and adapted to ingest a directory of PDF documents (https://github.com/wandb/edu/tree/main/llm-apps-course).


The whole application is built using langchain and gradio. 

## Setup
To test the application follow these steps:

1. Clone this repository: `git clone https://github.com/xmassmx/RAG.git`
2. Create a new python virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Launch src/app_local: `python src/app_local.py`

## Details

### Loading PDFs
For ingesting the PDF directory I used the `PyMuPDFLoader` function from `langchain.document_loaders`. This function creates separate Document objects for each page of the PDF. The Document object contains information about the page content, page number, source, and other metadata. 


### Chunking Data
The List of Document objects, corresponding to the pages of all the documents in the source documentation are then split into chunks of a predetermined size and overlap. In order to do this, the `RecursiveCharacterTextSplitter` from `langchain.text_splitter` is used with a default chunk_size of 3000 and chunk_overlap of 1000 tokens.


### Embedding model
The next step is to use an embedding model to convert the text into numerical representation that captures the semantic meaning of the text. The embedding model we use is the `OpenAIEmbeddings` from `langchain.embeddings`.


### Vectorstore
Once the documents are embedded, we need to store them. We store the embedding vectors in a vector database/ vectorstore. There are a few options to choose from but in general, all commonly used vector databases give similar functionalities, i.e. quick retrieval, and semantic similarity search. 
The vectorstore that I used is `Chroma` from `langchain.vectorstore`. 

### LLM Chain
The LLM chain that we use is `ConversationalRetrievalChain` which takes the user prompt as well as the chat history in order to answer the query. Once the user queries the RAG, the LLM chain first embeds the user query and retrieves top_k (where default k is 5) most semantically similar chunks from the vectorstore and inserts them into the prompt as the context. Currently I use the gpt-3.5-turbo model. 
