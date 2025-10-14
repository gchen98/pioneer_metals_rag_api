import os
import logging
#
## Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
#
#from langchain_core.prompts import PromptTemplate  # Updated import per deprecation notice
from langchain.chains import RetrievalQA
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_community.embeddings import HuggingFaceInstructEmbeddings  # New import path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader  # New import path
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # New import path

from langchain_ibm import WatsonxLLM

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Check for GPU availability and set the appropriate device for computation.
#DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE="cpu"

# Global variables
example_json = None
prompt_template = None
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings, prompt_template,example_json

    with open('example_json.txt','r') as file:
        example_json = file.readlines()[0]
    #logger.info(f"Example JSON {example_json}")

    prompt_template = f"""
    Make your best effort to populate values from the attached PDF into a human readable JSON string. Leave any JSON fields that you are uncertain about blank. For any values expressed as metric tons (MT), convert these values into kilograms (KG) by multiplying by 1000. Only return the JSON string in the response.  The JSON should have the same schema as the example: {example_json}
    """
    #logger.debug(f"Prompt is {prompt_template}")
    #exit(1)
       

    logger.info("Initializing WatsonxLLM and embeddings...")
    # credentials for Gary
    watsonx_API = "3nUB_JsQ5F-M-uONPbEdilh-0XGIAmlFR8PtyMwttwCN"
    project_id= "0845564a-9bcd-409e-b4b5-f9f8382cbb01"
    # Set up the credentials for accessing IBM Watson
    credentials = {
        'url': "https://us-south.ml.cloud.ibm.com",
        'apikey' : watsonx_API
    }
    # Set up parameters for the generation of text, including various settings such as the maximum and minimum number of new tokens to generate and the temperature
    params = {
        GenParams.MAX_NEW_TOKENS: 2000,
        GenParams.TEMPERATURE: 0.7,
    }
    # Set up the LLAMA2 model with the specified parameters and credentials
#    llama2_model = 'meta-llama/llama-2-13b-chat'
    llama2_model = 'meta-llama/llama-3-3-70b-instruct'
#    LLAMA2_model = Model(
#        model_id=llama2_model,
#        params=params,
#        credentials=credentials,
#        project_id=project_id)

    # Create a Watson LLM instance with the LLAMA2 model
    #llm_hub = WatsonxLLM(model=LLAMA2_model)
    llm_hub = WatsonxLLM(
        model_id=llama2_model,
        url="https://us-south.ml.cloud.ibm.com",
        apikey=watsonx_API,
        project_id=project_id,
        params = params
    )
    logger.debug("WatsonxLLM initialized: %s", llm_hub)


#    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
#    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
#    PROJECT_ID = "skills-network"
#
#    # Use the same parameters as before:
#    #   MAX_NEW_TOKENS: 256, TEMPERATURE: 0.1
#    model_parameters = {
#        # "decoding_method": "greedy",
#        "max_new_tokens": 256,
#        "temperature": 0.1,
#    }
#
#
#    # Initialize Llama LLM using the updated WatsonxLLM API
#    llm_hub = WatsonxLLM(
#        model_id=MODEL_ID,
#        url=WATSONX_URL,
#        project_id=PROJECT_ID,
#        params=model_parameters
#    )
#    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # Initialize embeddings using a pre-trained model to represent the text data.
    ### --> if you are using huggingFace API:
    # Set up the environment variable for HuggingFace and initialize the desired model, and load the model into the HuggingFaceHub
    # dont forget to remove llm_hub for watsonX

    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR API KEY"
    # model_id = "tiiuae/falcon-7b-instruct"
    #llm_hub = HuggingFaceHub(repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 600, "max_length": 600})
    
    #embeddings = HuggingFaceInstructEmbeddings(
    #    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    #    model_kwargs={"device": DEVICE}
    #)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={"device": DEVICE}
    )
    logger.debug("Embeddings initialized with model device: %s", DEVICE)

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain

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
    retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
    
    logger.debug("Chroma vector store initialized.")

    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.debug("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.warning("Could not retrieve collections from Chroma: %s", e)

    # Build the QA chain, which utilizes the LLM and retriever for answering questions. 
    context = example_json
    system_prompt = "{context} Do not return any other text other than a JSON formatted string"

    #system_prompt = (
    #    "Use the given context to answer the question. "
    #    "If you don't know the answer, say you don't know. "
    #    "Use three sentence maximum and keep the answer concise. "
    #    "Context: {context}"
    #)

#    prompt = ChatPromptTemplate.from_messages(
#        [
#            ("system", system_prompt),
#            ("human", "{input}"),
#        ]
#    )
    prompt = PromptTemplate(
        template=system_prompt,
        input_variables=['context','question']
    )

    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False,
        input_key="question",
        #chain_type_kwargs={"prompt": prompt}  # if you are using a prompt template, uncomment this part
    )
    question_answer_chain = create_stuff_documents_chain(llm_hub, prompt)
    #conversation_retrieval_chain = create_retrieval_chain(retriever,question_answer_chain)
    logger.info("RetrievalQA chain created successfully.")

# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    prompt = prompt_template

    logger.info("Processing prompt: %s", prompt)
    # Query the model using the new .invoke() method
    output = conversation_retrieval_chain.invoke({"question": prompt, "chat_history": chat_history})
    answer = output["result"]
    logger.debug("Model response: %s", answer)

    # Update the chat history
    chat_history.append((prompt, answer))
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    # Return the model's response
    return answer

# Initialize the language model
init_llm()
#logger.info("LLM and embeddings initialization complete.")
#pdffile='PACKING_LIST_9.8.pdf'
#process_document(pdffile)
#logger.info("PDF document processed.")
#process_prompt('')
#process_prompt(prompt_template)
#logger.info("PDF query processed.")
