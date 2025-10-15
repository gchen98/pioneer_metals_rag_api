import os
import logging

## Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma  # New import path
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader  # New import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ibm import WatsonxLLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

DEVICE="cpu"

# Global variables
prompt_template = None
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None
properties = {}

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings, prompt_template,properties
    with open('credentials.txt','r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split(': ')
            if len(tokens)==2:
               properties[tokens[0]] = tokens[1]
    logger.info(f"Properties {properties}")
    logger.info("Initializing WatsonxLLM and embeddings...")
    # Set up the credentials for accessing IBM Watson
    # Set up parameters for the generation of text, including various settings such as the maximum and minimum number of new tokens to generate and the temperature
    llm_hub = WatsonxLLM(
        model_id=properties['llm_model'],
        url=properties['watsonx.ai_url'],
        apikey=properties['apikey_value'],
        project_id=properties['project_id'],
        params = {
            GenParams.MAX_NEW_TOKENS: int(properties['max_new_tokens']),
            GenParams.TEMPERATURE: float(properties['temperature']),
        }
    )
    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceEmbeddings(
        model_name=properties['embedding_model'],
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings initialized with model device: %s", DEVICE)

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain
    # Initialize the language model
    init_llm()
    logger.info("Loading document from path: %s", document_path)
    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    logger.debug("Loaded %d document(s)", len(documents))

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    logger.info("Initializing Chroma vector store from documents...")
    db = Chroma.from_documents(texts, embedding=embeddings)
    logger.debug("Chroma vector store initialized.")

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    system_prompt = (
"""
You always answer the question in JSON format. The JSON structure and fields must match the JSON example provided by the user

Extract data from this PDF file: {context}

Question: {question}

"""
    )

    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=['question','context']
    )
    
    db_retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db_retriever,
        return_source_documents=False,
        input_key="question",
        chain_type_kwargs={"prompt": prompt}  # if you are using a prompt template, uncomment this part
    )
    logger.info("RetrievalQA chain created successfully.")

# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    logger.debug("Processing prompt: %s", prompt)
    # Query the model using the new .invoke() method
    output = conversation_retrieval_chain.invoke({"question": prompt})
    # not using a chat history for this.
    #output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    logger.debug("Model response: %s", answer)
    # Update the chat history
    chat_history.append((prompt, answer))
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))
    # Return the model's response
    return answer

