# Video RAG Assistant 🤖🎥

An intelligent Video Retrieval-Augmented Generation (RAG) system that allows users to query educational video content using natural language. This project transcribes video audio, converts it into searchable vector embeddings, and uses a local Large Language Model (LLM) to provide context-aware answers with timestamps.



## 🛠️ Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Transcription:** [OpenAI Whisper](https://github.com/openai/whisper) (Local)
- **Embeddings:** `bge-m3` via [Ollama](https://ollama.com/)
- **LLM:** `llama3.1:8b` or `deepseek-r1` via Ollama
- **Vector Search:** Cosine Similarity (Scikit-learn)
- **Processing:** FFmpeg for audio extraction

## 📂 Project Structure
- `app.py`: The Streamlit dashboard for the user interface.
- `processvideo.py`: Script to extract audio from MP4 files using FFmpeg.
- `mergechunks.py`: Combines small transcript segments into larger context blocks.
- `process_incoming.py`: Command-line testing script for query and retrieval.
- `embeddings.joblib`: Local storage for processed text and vector embeddings.
- `whisper/`: Local source code for the Whisper transcription engine.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Ollama** installed and the following models pulled:
```bash
ollama pull llama3.1:8b
ollama pull bge-m3
2. Installation
Clone the repository:

Bash

git clone [https://github.com/AlimpanMukherjee/RAG-PROJECT-1.git](https://github.com/AlimpanMukherjee/RAG-PROJECT-1.git)
cd RAG-PROJECT-1
Install Python dependencies:

Bash

pip install streamlit joblib numpy scikit-learn requests pandas
3. Pipeline Workflow
Follow these steps to process your own videos:

Audio Extraction: Place your videos in vdoResource/ and run:

Bash

python processvideo.py
Transcription: (Ensure stt.py is configured) to generate JSON transcripts.

Chunking: Merge transcript segments for better context:

Bash

python mergechunks.py
Launch UI: Start the RAG Assistant:

Bash

streamlit run app.py
🔍 How it Works
Retrieval: When you ask a question, the query is converted into a vector using the bge-m3 model.

Search: The system performs a cosine similarity search against the pre-computed embeddings.joblib.

Augmentation: The top 5 most relevant video chunks are injected into a custom prompt.

Generation: The local LLM generates a response that includes the Video Title and Exact Time Range where the answer can be found.

📜 License
MIT License - Feel free to use this for your own educational projects!


---

### How to add this to your project:
1. Open your terminal in the `RAGproject` folder.
2. Type `notepad README.md` and paste the text above.
3. Save and close.
4. Run these commands to update GitHub:
```powershell
git add README.md
git commit -m "Add professional README"
git push
