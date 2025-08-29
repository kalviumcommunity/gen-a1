import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

def call_gemini(prompt, temperature=0.7, top_p=0.95, top_k=40, stop=None):
    """
    Call the Gemini LLM with the given prompt and parameters.
    Args:
        prompt (str): The prompt to send to the LLM.
        temperature (float): Controls randomness of output.
        top_p (float): Nucleus sampling parameter (see video explanation).
        top_k (int): Top-k sampling parameter.
        stop (list): Stop sequences.
    Returns:
        str: The generated response.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, generation_config={
        'temperature': temperature,
        'top_p': top_p,  # Top-p (nucleus sampling) controls diversity of output
        'top_k': top_k,
        'stop_sequences': stop or []
    })
    return response.text
