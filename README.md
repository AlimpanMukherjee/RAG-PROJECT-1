# RAG-PROJECT-1
# Video RAG Assistant 🤖🎥

A Retrieval-Augmented Generation (RAG) system that allows you to ask questions about educational videos. It uses **Whisper** for transcription, **BGE-M3** for embeddings, and **Llama 3.1** via **Ollama** for generating answers.

## 🛠 Features
- **Video Processing:** Extracts audio and transcribes it using OpenAI's Whisper.
- **Semantic Search:** Uses cosine similarity to find the most relevant video segments.
- **Interactive UI:** Streamlit-based dashboard to query your video knowledge base.
- **Local LLM:** Fully private processing using Ollama.

## 🚀 Getting Started

### 1. Prerequisites
- **Ollama:** [Download here](https://ollama.com/)
- **FFmpeg:** Required for audio extraction.
- **Models:**
  ```bash
  ollama pull llama3.1:8b
  ollama pull bge-m3
