#from langchain import SagemakerEndpoint
from langchain.prompts import PromptTemplate
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import json
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
#from huggingface_hub import login
from dotenv import load_dotenv
load_dotenv()
#region = "us-east-1"
#endpoint_name = "*******************************************************"

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HF_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")  # Fetch the token from your environment
ENDPOINT_URL = "https://vqowi0abicmjh23l.us-east-1.aws.endpoints.huggingface.cloud"

def build_chain():

    # Sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})

    # Laod Faiss index
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True) #DB_FAISS_PATH kemungkinan yg bikin error

    # Default system prompt for the LLama2
    system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe with Indonesian Language.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""" #make sure


    # Langchain chain for invoking SageMaker Endpoint
    llm = HuggingFaceEndpoint(
        endpoint_url=ENDPOINT_URL,
        huggingfacehub_api_token=HF_API_TOKEN,
        timeout=500
    )
    
    """llm = HuggingFaceEndpoint(
        endpoint_url=ENDPOINT_URL,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        streaming=True #coba dari https://github.com/langchain-ai/langchain/issues/19685
    )"""
    


    def get_chat_history(inputs) -> str:
        res = []
        for _i in inputs:
            if _i.get("role") == "user":
                user_content = _i.get("content")
            if _i.get("role") == "assistant":
                assistant_content = _i.get("content")
                res.append(f"user:{user_content}\nassistant:{assistant_content}")
        return "\n".join(res)

    condense_qa_template = """
    Given the following conversation and a follow up question, rephrase the follow up question
    to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    standalone_question_prompt = PromptTemplate.from_template(
        condense_qa_template,
    )
    
    # Langchain chain for Conversation
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        condense_question_prompt=standalone_question_prompt,
        return_source_documents=False, #false to-do
        get_chat_history=get_chat_history
    )
    return qa


def run_chain(chain, prompt: str, history=[]):
    return chain({"question": prompt, "chat_history": history})

chain  = build_chain()
prompt = "what is Cuterebra?"

result = run_chain(chain=chain, prompt=prompt)

print("Result: ", result)


