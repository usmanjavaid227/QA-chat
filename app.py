import streamlit as st
import numpy as np
import torch
import faiss
from transformers import pipeline, DPRQuestionEncoder, DPRQuestionEncoderTokenizer

# ---------------------------
# Model and Pipeline Setup
# ---------------------------

# Summarization pipeline using BART
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Question Answering pipeline using DistilBERT
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load DPR question encoder and tokenizer
q_model_name = "facebook/dpr-question_encoder-single-nq-base"
q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
q_model = DPRQuestionEncoder.from_pretrained(q_model_name)

def encode_query(query):
    inputs = q_tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embedding = q_model(**inputs).pooler_output
    return embedding.squeeze().cpu().numpy()

# ---------------------------
# Dummy Data for Demonstration
# ---------------------------
try:
    all_segments  # Check if already defined
    index
except NameError:
    all_segments = [
        "This segment discusses gardening techniques and sustainable practices.",
        "Here we explore various agricultural methods and crop rotation benefits.",
        "This text covers organic farming and soil improvement tips.",
        "Insights on pest control and irrigation strategies are provided here.",
        "A discussion on modern greenhouse management and efficiency."
    ]
    # For demonstration, create dummy embeddings with dimension 768
    dummy_embeddings = np.random.rand(len(all_segments), 768).astype(np.float32)
    index = faiss.IndexFlatL2(768)
    index.add(dummy_embeddings)

# ---------------------------
# Chatbot Logic
# ---------------------------
def chatbot_response(query):
    query_lower = query.lower()
    # If the query starts with "tell me about" or "describe", use summarization logic
    if query_lower.startswith("tell me about") or query_lower.startswith("describe"):
        query_embedding = encode_query(query)
        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        k = 5  # Retrieve top 5 passages
        distances, indices = index.search(query_embedding, k)
        retrieved_texts = [all_segments[i] for i in indices[0] if i < len(all_segments)]
        combined_text = " ".join(retrieved_texts)
        summary = summarizer(combined_text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        return summary
    else:
        # Otherwise, assume it's a direct question for QA
        query_embedding = encode_query(query)
        query_embedding = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        k = 5
        distances, indices = index.search(query_embedding, k)
        context = " ".join([all_segments[i] for i in indices[0] if i < len(all_segments)])
        answer = qa_pipeline(question=query, context=context)
        return answer['answer']

# Test the chatbot logic (this line will print to the console when running locally)
test_query = "Tell me about sustainable agriculture practices."
print("Chatbot response:", chatbot_response(test_query))

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Agricultural eBooks Chatbot", layout="centered")

# Custom CSS styling for a nice look
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}
h1 {
    font-size: 2.5rem;
    font-weight: bold;
    color: #333;
    text-align: center;
}
.stButton>button {
    background-color: #000000;
    color: #ffffff;
    padding: 12px 24px;
    border: 2px solid #000000;
    border-radius: 8px;
    font-size: 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Agricultural eBooks Chatbot")
st.markdown("Enter your query below. For example, **tell me about ...** for a summary or ask a direct question for QA.")

# Input field for the query
query = st.text_input("Your Query", placeholder="Enter your query here...")

# When the user clicks 'Submit', process the query
if st.button("Submit"):
    if query:
        with st.spinner("Processing..."):
            response = chatbot_response(query)
        st.markdown("### Chatbot Response")
        st.write(response)
    else:
        st.error("Please enter a query before submitting.")
