import os
from llm_call import call_gemini

PROMPT_PATH = os.path.join(os.path.dirname(__file__), '../prompts/zero_shot_prompt.txt')
with open(PROMPT_PATH, 'r') as f:
    PROMPT_TEMPLATE = f.read()

def zero_shot_answer(query):
    prompt = PROMPT_TEMPLATE.format(query=query)
    return call_gemini(prompt)

if __name__ == '__main__':
    user_query = input('Ask the AI Sports Coach: ')
    answer = zero_shot_answer(user_query)
    print('\nAI Coach says:', answer)
