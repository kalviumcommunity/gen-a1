import streamlit as st
import requests

st.set_page_config(page_title="AI Sports Coach Demo", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ AI Sports Coach (RAG Demo)")

st.markdown("""
Ask any sports, fitness, or training question and get personalized, grounded advice from the AI Sports Coach!
""")

query = st.text_input("Enter your question:", "How can I improve my upper body strength?")

temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.01)
top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.95, 0.01)
top_k = st.slider("Top-k", 1, 100, 40, 1)

if st.button("Ask Coach"):
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                "http://localhost:8000/ask",
                json={
                    "query": query,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "stop": None
                },
                timeout=30
            )
            if resp.status_code == 200:
                st.success(resp.json()["response"])
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

st.markdown("---")
st.caption("Demo powered by Gemini, RAG, and Streamlit.")
