import gradio as gr
import httpx

API_BASE_URL = "http://127.0.0.1:8000"


def ingest_files(files: list[str]) -> str:
    if not files:
        return "No files selected."
    payload = []
    opened = []
    try:
        for path in files:
            fh = open(path, "rb")
            opened.append(fh)
            payload.append(("files", (path.split("/")[-1].split("\\")[-1], fh, "application/octet-stream")))
        resp = httpx.post(f"{API_BASE_URL}/ingest", files=payload, timeout=120.0)
        resp.raise_for_status()
        results = resp.json()["results"]
        lines = [
            f"{r['filename']}: {'ok, ' + str(r['chunks_added']) + ' chunks' if r['ok'] else 'FAILED - ' + r['error']}"
            for r in results
        ]
        return "\n".join(lines)
    except httpx.HTTPError as exc:
        return f"Ingest failed: {exc}"
    finally:
        for fh in opened:
            fh.close()


def ask_question(question: str, history: list) -> tuple[str, list]:
    if not question.strip():
        return "", history

    try:
        resp = httpx.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=60.0)
        resp.raise_for_status()
        body = resp.json()
    except httpx.HTTPError as exc:
        history = history + [(question, f"Error contacting API: {exc}")]
        return "", history

    answer = body["answer"]
    if body["citations"]:
        sources = "\n".join(
            f"  - {c['source_path']} (score {c['score']:.2f})" for c in body["citations"]
        )
        answer = f"{answer}\n\nSources:\n{sources}"
    history = history + [(question, answer)]
    return "", history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="RAG Platform") as demo:
        gr.Markdown("# RAG Platform")
        with gr.Tab("Ingest"):
            file_input = gr.File(file_count="multiple", type="filepath", label="Upload .txt / .md / .pdf")
            ingest_btn = gr.Button("Ingest")
            ingest_output = gr.Textbox(label="Ingest results", lines=8)
            ingest_btn.click(fn=ingest_files, inputs=file_input, outputs=ingest_output)
        with gr.Tab("Ask"):
            chatbot = gr.Chatbot()
            question_box = gr.Textbox(label="Question", placeholder="Ask about your ingested documents...")
            question_box.submit(fn=ask_question, inputs=[question_box, chatbot], outputs=[question_box, chatbot])
    return demo


if __name__ == "__main__":
    build_ui().launch()
