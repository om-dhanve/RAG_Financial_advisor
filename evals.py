#Using RAGAS For evaluations : 
from ragas import evaluate
import pandas as pd
from datasets import Dataset
from models import ChatRequest, ChatResponse
from dotenv import load_dotenv
from config import Settings
from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from response_generation import chat_with_rag
from ragas.metrics import _faithfulness,_context_recall,_context_precision #faithfulness, answer_relevancy, context_precision, context_recall
import os
import google.generativeai as geminiai
from ragas.llms import llm_factory
load_dotenv()
settings = Settings()


geminiai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Create client
llm1 = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0,timeout=240)#("gemini-2.0-flash")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # or "models/embedding-001" for Gemini embeddings
    google_api_key=os.environ["GOOGLE_API_KEY"]
)

# dataset = 
queries = [
            "What are the steps to update my occupation details in HDFC NetBanking, and what information can I modify?", 
            "How do I request my CIBIL score from HDFC NetBanking, and what should I know before doing so?",
            "Compare SBI Car loan facility with other banks?",
            "What is the path to see my taxes paid and my CIBIL score on HDFC Net banking?",
            "What is the minimum deposit amount for a tax‑saving FD in SBI Bank?",
]

ground_truths = [   
        "To update your occupation details in HDFC NetBanking, follow this path: Log in to HDFC Bank NetBanking > Dashboard > Hamburger Menu > My Profile > Update Occupation. From this page, you can modify the following details: Profession, Company Type, Source of Funds, Gross Annual Income, and Residence Type. Note that some fields may have restrictions based on your customer profile, so verify each field before submission.",
        "You can request your Credit Report (CIBIL score) from HDFC NetBanking by following this path: Log in to HDFC Bank NetBanking > Dashboard > Hamburger Menu > My Profile > Request CIBIL Score. Before requesting, be aware that CIBIL charges Rs 550 for this report, which will be deducted from your account. The request is processed through the NetBanking interface, but you should expect to receive the actual report from CIBIL within a few business days.",
        "The following are the advantages of SBI Car loan process compared to others : There is total transparency with regards to the rate of interest and the fees charged by us. No Pre-payment charges. Competitive rate of interest. 100 Percent on road finance on selected models We levy interest on daily reducing balance, unlike the flat rate of interest or interest based on annual reducing balance method used by several other financiers. We provide finance for one-time road tax, registration fee and insurance premium also.We do not charge any advance EMIs.We offer loans for the longest tenors (96 months for Electric Cars). We provide finance for both new vehicles as well as Certified Pre-owned cars.",
        "Path: Log in to HDFC Bank NetBanking > Dashboard > Hamburger > My Profile > Request CIBIL Score. (Please note that you will be charged Rs 550 by CIBIL for the report). You can view your tax Summary/Certificate/Report at below path:\nLog in to HDFC Bank NetBanking > Dashboard > Hamburger > Tax Services > Tax Deducted at Source.",
        "I do not have context on the minimum deposit amount for SBI bank."
    ]

results = []
contexts = []

for q in queries:
    result = chat_with_rag(ChatRequest(message=q))

    results.append(result.response)
    sources = result.context
    contents = []
    for i in sources:
        contents.append(i)
    contexts.append(contents)

data_dict = {
    "user_input": queries,
    "response": results,
    "retrieved_contexts": contexts,
    "reference": ground_truths,
}

dataset = Dataset.from_dict(data_dict)
score = evaluate(dataset=dataset,metrics=[_faithfulness,_context_precision,_context_recall],llm=llm1,batch_size=2)
print(score)
score_df = score.to_pandas()
score_df.to_csv("Eval.csv",encoding="utf-8",index=False)
