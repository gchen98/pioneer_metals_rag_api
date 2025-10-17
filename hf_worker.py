import os
import logging
import json

from langchain import hub

from langchain_community.document_loaders import PyPDFLoader  # New import path
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # New import path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


from doctr.io import DocumentFile
from doctr.models import ocr_predictor



DEVICE="cpu"

## Configure logging
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init_hf():
    global chat_model,embeddings,ocr_model
    # load all the properties from file
    properties = {}
    with open('hf_properties.txt','r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split(': ')
            if len(tokens)==2:
               properties[tokens[0]] = tokens[1]
    model_name = properties['llm_model']
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
    logger.info("Initialized HF endpoint...")
    chat_model = ChatHuggingFace(llm=llm)
    logger.info("Initialized HF chat model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=properties['embedding_model'],
        model_kwargs={"device": DEVICE}
    )
    logger.info("Embeddings initialized with model device: %s", DEVICE)
    ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
    logger.info("OCR model initialized ")


def run_ocr(document_path):
    # PDF
    doc = DocumentFile.from_pdf(document_path)
    # Analyze
    logger.info("Launch OCR...")
    result = ocr_model(doc)
    logger.info(f"OCR completed on {document_path}")
    #logger.debug(f"Contents {result.render()}")
    document = Document(
    page_content=result.render(), metadata={"source": document_path}
    )
    return [document]

def load_document(document_path):
    global db,chunks
    # Load the document
    #loader = PyPDFLoader(document_path)
    #document = loader.load()
    document = run_ocr(document_path)
    logger.info("Loaded document from path: %s", document_path)
    logger.debug(f"Document {document}")
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=512)
    chunks = text_splitter.split_documents(document)
    logger.info("Document split into %d text chunks", len(chunks))
    for text in chunks:
        logger.debug(f"\nCHUNK: {str(text)}")
    db = Chroma.from_documents(chunks, embedding=embeddings)
    logger.info("Chroma vector store initialized.")
    # Optional: Log available collections if accessible (this may be internal API)
    try:
        collections = db._client.list_collections()  # _client is internal; adjust if needed
        logger.info("Available collections in Chroma: %s", collections)
    except Exception as e:
        logger.error("Could not retrieve collections from Chroma: %s", e)

def format_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)

def make_prompt():
    system_prompt_json = """
        You always answer the question in JSON format in a single line. The returned JSON must match the structure and field names of the JSON provided by the user. Populate only blank JSON fields.  Always enclose JSON values in double quotes. For any values expressed as metric tons (MT), when possible first convert these values into kilograms (KG) by multiplying by 1000 before populating the JSON.  Extract values from the PDF {context}.
        """
    
    template = ChatPromptTemplate(
        [
            ("system",system_prompt_json),
            ("human","JSON example is {question}")
        ]
    )
    return template

def load_example_json(path):
    with open(path,'r') as file:
        jsonstr = file.readline().strip()
    return jsonstr

def get_response(query):
    prompt = make_prompt()
    prompt.pretty_print()

    #prompt = hub.pull("rlm/rag-prompt")
    #prompt.pretty_print()
    #print(type(prompt))
    #exit(1)

    logger.info("Loaded prompt")
    qa_chain = (
        {
            "context":db.as_retriever() | format_chunks ,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )   
    logger.info("Created QA chain")
    response = qa_chain.invoke(query)
    return response


def test_module():
    #run_ocr("samples/audubon.pdf")
    #run_ocr("samples/PACKING_LIST_9.8.pdf")
    #exit(1)
    load_document("samples/audubon.pdf")
    #load_document("samples/PACKING_LIST_9.8.pdf")
    jsonstr = load_example_json('samples/example_json.txt')
    response = get_response(jsonstr)
    logger.info(f"Response: {response}")

init_hf()
#test_module()
