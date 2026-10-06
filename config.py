from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    #Configuring .env file : 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_name : str = "Rag_Financial_advisor"

    #LLM 
    google_api_key : str = ""
    temperature : float = 0.3
    top_k : int = 4
    llm_model : str = "gemini-3.5-flash" 

    #Embeddings 
    hf_token: str = ""
    embedding_model : str = "gemini-embedding-2"

    #Vector DB
    supabase_database_url : str = ""

    #Data sources : For every change in data source metadata should be manually changed too
    html_files_path :str = ".\\Finance_Advisor_IngestionDoc\\htmlfiles\\FAQ RTGS NEFT - Faq's.html" 
    txt_file_path : str = ".\\Finance_Advisor_IngestionDoc\\txtfiles\\sbi.bank.in faq-car-loan.txt"

    #RAG
    chunk_size : int = 500
    chunk_overlap : int = 50

    #Langsmith | Tracing
    langsmith_tracing : bool = False #Setting false as default
    langsmith_endpoint : str = "" 
    langsmith_api_key: str = ""
    langsmith_project : str = ""


settings = Settings()