

# SOFTDOC - Software Documentation Assistant

**SOFTDOC** is a Streamlit-based web application designed to help users understand software documentation by answering queries based on uploaded PDFs.

## Features

- PDF Upload
- Chatbot interaction for answering software-related questions
- Contextual understanding using LLM and NLP techniques
- Chat history to maintain continuity

## Installation and Setup

1. Clone this repository and navigate to the `SOFTDOC` folder:
   ```bash
   git clone https://github.com/shreeramdrao/FWC-Projects.git
   cd SOFTDOC
   ```
2. Install the required libraries:
   ```bash
   pip install streamlit PyPDF2 sentence-transformers faiss-cpu openai
   ```
3. Set your OpenAI API key in the code:
   ```python
   openai.api_key = 'YOUR_API_KEY'
   ```
4. Run the project:
   ```bash
   streamlit run app.py
   ```

