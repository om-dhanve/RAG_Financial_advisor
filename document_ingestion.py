#Document ingestion 

## Using the BSHTMLLoader for loading .html files
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import BSHTMLLoader,TextLoader
from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langsmith import traceable
from config import settings
from dotenv import load_dotenv
from langchain_postgres import PGVector
load_dotenv()

#Initializing document loader, Text splitter and embedding model
# HTML_loader = BSHTMLLoader(file_path=None,open_encoding='utf-8',get_text_separator="")
# TXT_Loader = TextLoader(file_path=None,encoding='utf-8')
splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,chunk_overlap=settings.chunk_overlap)
embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model,model_kwargs={"token":settings.hf_token})

@traceable(name="document_loading",enabled=True)
def load_documents(file_path,bank,doc_type,doc_title,source):
    """Loading documents with associated metadata"""
    docs = []

    if doc_type == "HTML_file":
        doc_loader = BSHTMLLoader(file_path=file_path,open_encoding='utf-8',get_text_separator="")
    elif doc_type == "TXT_file":
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
                           collection_name="RAG_Bank_FAQ's",create_extension=True)
    print(f"Vector store created.")
    return vectorstore

def load_to_vectorstore(vectorstore,split_documents):

    for doc in split_documents:
        ids = vectorstore.add_documents([doc])
        print(f"Document added to Supabase {ids[0]}")

if __name__ == "__main__" :
    splitdocs = load_documents(file_path=settings.txt_file_path,bank="SBI",doc_type="TXT_file",doc_title="SBI ATM Services FAQ",source="https://sbi.bank.in/web/faq-s/faq-atm-services")
    vectorestore = vectorstore_creation()
    print(f"Split docs {"---------------++--------------\n\n\n".join([doc.page_content for doc in splitdocs])}")
    print(f"Split docs metadata {splitdocs[2].metadata}")
    load_to_vectorstore(vectorestore,splitdocs)


## ---- only enable when embedding process is to be redone
# load_to_vectorstore(vectorestore)

    

