from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from passlib.context import CryptContext
import uvicorn
import json
#from langchain import SagemakerEndpoint
from langchain.prompts import PromptTemplate
#from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings #error di sini (_community)
from langchain_community.vectorstores import FAISS #error di sini (_community)
import re
from langchain_config import run_chain, chain  # Import the chain from langchain_config.py
from fastapi.responses import JSONResponse
import os
from langchain_config import HuggingFaceEndpoint 


#region = "us-east-1"
#endpoint_name = "**********************************************"

DB_FAISS_PATH = 'vectorstore/db_faiss'

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable session management
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# Simulated database for users
fake_users_db = []

# User model
class User:
    def __init__(self, username: str, hashed_password: str):
        self.username = username
        self.hashed_password = hashed_password

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Hashed passwords for the sample users (You should hash real passwords before storing them)
hashed_password_user1 = pwd_context.hash("admin@123")  # Replace with the actual hashed password
hashed_password_user2 = pwd_context.hash("user@123")   # Replace with the actual hashed password

# Create sample users
fake_users_db.append(User(username="admin", hashed_password=hashed_password_user1))
fake_users_db.append(User(username="user", hashed_password=hashed_password_user2))

# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

# Function to get a user by username
def get_user(username: str):
    for user in fake_users_db:
        if user.username == username:
            return user

def build_chain():

    # Sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})

    # Laod Faiss index
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    # Custom ContentHandler to handle input and output to the SageMaker Endpoint
    llm = HuggingFaceEndpoint(
        endpoint_url="https://vqowi0abicmjh23l.us-east-1.aws.endpoints.huggingface.cloud",
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
      )  # Set your Hugging Face API token as an environment variable
    
    # Langchain chain for Conversation
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        # chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True #try to change to False later
    )
    return qa


def run_chain(chain, prompt: str):    
    return chain({"query": prompt})

chain = build_chain()

# revisi 3 kata gemini suruh remove def process_answer(answer)
"""def process_answer(answer):
    answer_list = answer.split("\n")
    answer_list.pop(0)
    answer = "\n".join(answer_list)
    answer1 = answer.split("Question:")[0]
    print("Answer1: ", answer1)
    answer2 = answer.split("Helpful Answer:")[1].split("Context:")[0]
    print("Answer2: ", answer2)
    final_answer = answer1 + '\n' + answer2

    return final_answer"""

# Login route
@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = get_user(username)
    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # For simplicity, store user information in session and redirect to a protected page
    request.session['user'] = user.username
    response_data = jsonable_encoder(json.dumps({"msg": "Success",}))
    res = Response(response_data)
    return res

# Logout route
@app.get("/logout")
async def logout(request: Request):
    # Remove user information from session
    request.session.pop('user', None)
    return RedirectResponse(url="/")

# Chat route (protected)
@app.get("/chat")
async def chat(request: Request):
    # Check if user is logged in
    user = request.session.get('user')
    if user is None:
        # Redirect to login if not logged in
        return RedirectResponse(url="/")
    return templates.TemplateResponse("chat.html", {"request": request, "user": user})

# Root route
@app.get("/")
async def read_root(request: Request):
    # Check if user is logged in
    user = request.session.get('user')
    if user is None:
        # Display login page if not logged in
        return templates.TemplateResponse("login.html", {"request": request})
    else:
        # Redirect to chat if logged in
        return RedirectResponse(url="/chat")

"""@app.post("/chat_response")
async def chat_resonse(request: Request, prompt: str = Form(...)):
    result = run_chain(chain=chain, prompt=prompt)
    answer = process_answer(result['result'])
    source_documents = result['source_documents']
    source_documents_list = []
    page_number_list = []
    for doc in source_documents:
        source_doc = doc.metadata['source']
        page_number = doc.metadata['page']
        if source_doc not in source_documents_list:
            source_documents_list.append(source_doc)
            page_number_list.append(page_number)

    response_data = jsonable_encoder(json.dumps({"answer": answer, "source_documents_list": source_documents_list, "page_number_list": page_number_list}))
    res = Response(response_data)
    return res"""

# revisi 1 @app.post("/chat_response")
"""@app.post("/chat_response")
async def chat_response(request: Request, prompt: str = Form(...)):
    # Get chat history from the session, if available
    chat_history = request.session.get("chat_history", [])

    # Get response from the LLM chain
    result = run_chain(chain, prompt, chat_history)

    # Append the user's message and the model's response to the chat history
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": result["answer"]})

    # Store the updated chat history in the session
    request.session["chat_history"] = chat_history

    # Prepare the response data
    response_data = jsonable_encoder(
        json.dumps(
            {
                "answer": result["answer"],
                "source_documents_list": [
                    doc.metadata["source"] for doc in result["source_documents"]
                ],
            }
        )
    )
    res = Response(response_data)

    return res"""

# revisi 2 @app.post("/chat_response")
@app.post("/chat_response")
async def chat_response(request: Request, prompt: str = Form(...)):
    try:
        result = run_chain(chain, prompt) # No need to pass chat_history explicitly
        print("Result from LLM:", result)
        if 'result' in result:
            answer = result['result']
        elif 'answer' in result:
            answer = result['answer']
        else:
            raise KeyError('Neither "result" nor "answer" key found in LLM response')

        # If the answer key doesn't exist, try a different key:
        # answer = result["some_other_key"]["generated_text"]
        
        #answer = result['answer']
        #answer = result['output']
        #answer = result["response"]["text"]  # Extract the answer from result
        source_documents = result["source_documents"]

        response_data = jsonable_encoder(
            json.dumps(
                {
                    "answer": answer,
                    "source_documents_list": [
                        doc.metadata["source"] for doc in source_documents
                    ],
                }
            )
        )
        return JSONResponse(content=response_data)
        

    except Exception as e:  # Catch any potential exceptions
        print(f"Error getting response from LLM: {type(e).__name__}: {e}") # Log the exception
        return JSONResponse(
            content={"error": "Sorry, couldn't get a response at this time."},
            status_code=500,
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)
