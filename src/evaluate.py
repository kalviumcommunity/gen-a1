import json
from rag_pipeline import RAGPipeline

with open('../evaluation/test_dataset.json') as f:
    test_data = json.load(f)
with open('../prompts/system_prompt.txt') as f:
    prompt_template = f.read()

rag = RAGPipeline()

for item in test_data:
    query = item['query']
    expected = item['expected']
    actual = rag.run(query, prompt_template)
    print(f"Query: {query}\nExpected: {expected}\nActual: {actual}\n{'-'*40}")
