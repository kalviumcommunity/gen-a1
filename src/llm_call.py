import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

def count_tokens(text):
    """
    Simple token counter (splits on whitespace). For demo purposes only.
    """
    return len(text.split())

def call_gemini(prompt, temperature=0.7, top_p=0.95, top_k=40, stop=None):
    """
    Call the Gemini LLM with the given prompt and parameters.
    Args:
        prompt (str): The prompt to send to the LLM.
        temperature (float): Controls randomness of output.
        top_p (float): Nucleus sampling parameter (see video explanation).
        top_k (int): Top-k sampling parameter (see video explanation).
        stop (list): Stop sequences.
    Returns:
        str: The generated response.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, generation_config={
        'temperature': temperature,
        'top_p': top_p,  # Top-p (nucleus sampling) controls diversity of output
        'top_k': top_k,  # Top-k limits sampling to the k most likely tokens
        'stop_sequences': stop or []
    })
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(response.text)
    total_tokens = prompt_tokens + response_tokens
    print(f"[Token Log] Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    return response.text
