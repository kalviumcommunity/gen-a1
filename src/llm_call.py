import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

def call_gemini(prompt, temperature=0.7, top_p=0.95, top_k=40, stop=None):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, generation_config={
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'stop_sequences': stop or []
    })
    return response.text
