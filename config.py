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
    llm_model : str = "gemini-3.1-flash-lite" 

    #Embeddings 
    hf_token: str = ""
    embedding_model : str = "BAAI/bge-small-en-v1.5"

    #Vector DB
    supabase_database_url : str = ""

    #Data sources
    html_files_path_1 :str = "" #File path for embedding docs
    txt_file_path : str = ""

    #RAG
    chunk_size : int = 500
    chunk_overlap : int = 50

    #Langsmith | Tracing
    langsmith_tracing : bool = False #Setting false as default
    langsmith_endpoint : str = "" 
    langsmith_api_key: str = ""
    langsmith_project : str = ""


settings = Settings()