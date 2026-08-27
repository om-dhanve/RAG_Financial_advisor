#User interface for calling the RAG system

import gradio as gr
import re
from response_generation import chat_with_rag
from models import ChatRequest

# =================Theme================
# ======================================

import gradio as gr

theme = gr.themes.Ocean(
    spacing_size="sm",
    radius_size="lg",
    font=[gr.themes.GoogleFont('Google sans'), 'IBM Plex Sans', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('Inter'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    body_text_color='*neutral_900',
    body_text_color_subdued='*neutral_800',
    body_text_color_subdued_dark='*neutral_600',
    body_text_weight='500',
    embed_radius='*radius_md'
)

#==================Helper func============
#=========================================

def clean_text(text):
    text = re.sub(r"\n+"," ",text)
    text = text.replace("\xa0"," ")
    text = re.sub(r"\s+"," ",text)
    text = text.strip()
    return text

def invoke_chatwithrag(query):
    question = ChatRequest(message=query)
    result = chat_with_rag(question)
    if result.sources : 
        sources_display = "\n".join(
            f"{src.get('doc_title', 'Unknown bank')} - "
            f"{src.get('source','no URL')} | "
            for src in result.sources
        )
        source_count = len(result.sources)
    else : 
        sources_display = "No sources found"
        source_count = 0

    status_line = (
        f"Returned in {result.response_time:.2f}s | {result.model_name} | "
        f"{source_count} Relevant sections found{'s' if source_count != 1 else ''}"
    )
    cleaned_context = [clean_text(c) for c in result.context]
    display_context = "\n\n".join(f" {i+1} - {c}" for i,c in enumerate(cleaned_context))
    return result.response ,display_context, sources_display, status_line  #f"{result.sources["doc_title"]} : {result.sources["source"]}",f"Returned in {result.response_time:.2f}ms | {result.model_name} | {result.sources["source"]} source documents"


with gr.Blocks() as app:
    gr.Markdown("## Autonomous Financial RAG System ",scale=4)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Please input your question below :",scale=2)
            query_input = gr.Textbox(label="User question",max_lines=15,placeholder="What is the interest rate on FD in HDFC bank?",lines=4,scale=1,min_width=60)
            submit_button = gr.Button(variant="primary",value="Get Answer",scale=1)

            gr.Markdown("### Answer :")
            answer_output = gr.Markdown(value="Response to question will appear here ",scale=1)

        with gr.Column(scale=1):

            gr.Markdown("### Context Retrieved from Source Docs : ")
            source_docs = gr.Textbox(label="Context Docs",placeholder="Your retrieved context will appear here.",lines=4)
            gr.Markdown("### Sources :")
            sources_output = gr.Markdown(label="RAG Output",value="Links to sources will appear here",scale=1)
            gr.Markdown("### Runtime metrics")
            latency = gr.Markdown("Response time | Model name | No. of source docs",value="Runtime metrics")

            submit_button.click(
                fn=invoke_chatwithrag,
                inputs=(query_input),
                outputs=[answer_output,source_docs,sources_output,latency]
            )

if __name__ == "__main__":
    app.launch(share=False,theme=theme)