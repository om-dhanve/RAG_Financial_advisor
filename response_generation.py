# Embedding user query, retrieval from db and generating response from LLM

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_postgres import PGVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv
from config import Settings
from models import ChatRequest,ChatResponse,ErrorRespone,MetricsResponse
from document_ingestion import vectorstore_creation
import uuid,time
settings = Settings()

#Model config
llm = GoogleGenerativeAI(model=settings.llm_model,temperature=settings.temperature)
embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model,model_kwargs={"token":settings.hf_token})

@traceable(name="Chat_with_RAG",run_type="chain")
def chat_with_rag(user_query : ChatRequest) ->ChatResponse:

    vectorstore = vectorstore_creation()
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":settings.top_k}) 
    start_time=time.time()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            """You are a banking and finance Q & A assitant. 
            Include all relevant details from the context that answer the question and is directly relevant to the question
            and display to the user.
            Prefer completeness over brevity. A correct multi step answer is better than an abstracted one-liner answer.
            Only omit the context that is genuinely irrelevant.  
            If you cannot answer from the context, say that i dont have context.\n\n
            Context:\n{context}"""
        ),
        (
            "user","{question}"
        )
    ])

    #Formatting retrieved documents with spacing 
    def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs]) 

    def format_sources(docs):
            if not docs:
                  return None
            return [doc.metadata for doc in docs]

    def format_context_docs(docs):
          if not docs:
                return None
          return [doc.page_content for doc in docs]

    fetched_sources = vectorstore.similarity_search(query=user_query.message,k=3)
    documents = format_sources(fetched_sources)
    context_docs = format_context_docs(fetched_sources)

    #Defining RAG Chain
    rag_chain = (
            {"context" : retriever | format_docs, "question":RunnablePassthrough()}
            | prompt
            | llm 
            | StrOutputParser()
    )

    #invoking the rag chain for questions
    results = rag_chain.invoke(user_query.message)
    return ChatResponse(
         response=results,
         sources= documents,
         context= context_docs,
         response_time=time.time()-start_time,
         model_name= settings.llm_model,
         thread_id=str(uuid.uuid4())
         )
    

if __name__ == "__main__" :

    print("🖥️ Initializing RAG-Financial Advisor...")
    question = input(f"Enter your question : ")
    user_question = ChatRequest(message=question)
    response = chat_with_rag(user_question)
    print("\n\n--------- Response ------------\n",response.response)
