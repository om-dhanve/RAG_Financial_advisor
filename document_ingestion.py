#Document ingestion 

## Using the BSHTMLLoader for loading .html files
# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import BSHTMLLoader,TextLoader
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langsmith import traceable
from config import settings
from dotenv import load_dotenv
from langchain_postgres import PGVector
import time
load_dotenv()

splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,chunk_overlap=settings.chunk_overlap)
embedding = GoogleGenerativeAIEmbeddings(model=settings.embedding_model,output_dimensionality=768)

@traceable(name="document_loading",enabled=True)
def load_documents(file_path,bank,doc_type,doc_title,source):
    """Loading documents with associated metadata"""
    docs = []

    if doc_type == ".html":
        doc_loader = BSHTMLLoader(file_path=file_path,open_encoding='utf-8',get_text_separator="")
    elif doc_type == ".txt":
        doc_loader = TextLoader(file_path=file_path,encoding='utf-8')
    else :
        print("Invalid document type")

    docs = doc_loader.load()
    print("✔️ Documents are loaded...")

    for doc in docs:
        doc.metadata.update(
            {
                "bank":bank,
                "doc_type" : doc_type,
                "doc_title": doc_title,
                "source" : source
            }
        )
        print("Metadata updated...")

    split_docs = splitter.split_documents(docs)
    print("✔️ Documents split in chunks...")
    print(f"✔️ The number of chunks {len(split_docs)}")

    return split_docs

def vectorstore_creation() : 

    vectorstore = PGVector(embeddings=embedding,
                           connection=settings.supabase_database_url,
                           collection_name="RAG_Bank_FAQ's Gemini-embedding",create_extension=True,
                           collection_metadata={"embeddings" : "RAG_Bank_FAQ's Gemini-embedding"})
    print(f"Vector store created.")
    return vectorstore

def load_to_vectorstore(vectorstore,split_documents):
    count = 0
    for doc in split_documents:
        ids = vectorstore.add_documents([doc])
        print(f"Document {count} added to Supabase {ids[0]}")
        count += 1

def batch_embed(vectorstore,split_documents,batch_size : int):
    try :
        
        for i in range (0,len(split_documents),batch_size) :
            batch  = split_documents[i:i+batch_size]
            print(f"Size of batch : {len(batch)}")
            ids = vectorstore.add_documents(batch)
            time.sleep(25)

        print(f"Sample batch {batch[:300]}")

        # print(f"Documents added to the vector store {ids}")
        return f"Documents embedded in batch of {batch_size}"

    except Exception as e : 
        return f"Error occurred {e}" 
    
if __name__ == "__main__" :

    vectorestore = vectorstore_creation()
    splitdocs = load_documents(file_path=settings.txt_file_path,bank="SBI",doc_type=".txt",doc_title="SBI Car loan FAQs",source="https://sbi.bank.in/web/faq-s/faq-car-loan")
    print(f"Split docs {"\n\n---------------++--------------\n\n".join([doc.page_content for doc in splitdocs])}")
    print(f"Split docs metadata {splitdocs[0].metadata}")
    batchresult = batch_embed(vectorstore=vectorestore,split_documents=splitdocs,batch_size=20)
    # print(batchresult)
    # load_to_vectorstore(vectorestore,splitdocs) ## ---- only enable when embedding process is to be redone



    

