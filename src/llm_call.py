import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key = GEMINI_API_KEY)

def count_tokens(text):
    """
    Simple token counter (splits on whitespace). For demo purposes only.
    """
    return len(text.split())


def call_gemini(prompt, temperature=0.7, top_p=0.95, top_k=40, stop=None, structured=False):
    """
    Call the Gemini LLM with the given prompt and parameters.
    Optionally request structured (JSON) output and parse it.
    Args:
        prompt (str): The prompt to send to the LLM.
        temperature (float): Controls randomness of output. Higher values (e.g., 1.0) make output more random, lower values (e.g., 0.2) make it more deterministic. (See video explanation.)
        top_p (float): Nucleus sampling parameter (see video explanation).
        top_k (int): Top-k sampling parameter (see video explanation).
        stop (list): Stop sequences.
        structured (bool): If True, request and parse JSON output.
    Returns:
        str or dict: The generated response, as JSON if structured=True and parsing succeeds.
    """
    if structured:
        prompt += "\n\nRespond ONLY in valid JSON format as: {\"answer\": <your answer>, \"steps\": [<step1>, <step2>, ...]}"
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt, generation_config={
        'temperature': temperature,
        'top_p': top_p,  # Top-p (nucleus sampling) controls diversity of output
        'top_k': top_k,  # Top-k limits sampling to the k most likely tokens
        'stop_sequences': stop or []
    })
    output = response.text
    prompt_tokens = count_tokens(prompt)
    response_tokens = count_tokens(output)
    total_tokens = prompt_tokens + response_tokens
    print(f"[Token Log] Prompt: {prompt_tokens}, Response: {response_tokens}, Total: {total_tokens}")
    if structured:
        import json
        try:
            return json.loads(output)
        except Exception:
            print("[Warning] Could not parse structured output as JSON. Returning raw text.")
            return output
    return output
