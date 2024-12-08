import openai
from config import CONFIG

PROMPT = """
You are a helpful assistant that can answer questions.
Rules:
- Reply with the answer only and nothing but the answer.
- Say "I don't know" if you don't know the answer.
- Use the provided context.
"""

class QuestionAnsweringBot:
    def __init__(self, llm_key):
        openai.api_key = llm_key

    def generate_answer(self, prompt):
        try:
            completion = openai.ChatCompletion.create(
                model=CONFIG['OPENAI_ENGINE'],
                messages=[
                    # {"role": "system", "content": PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=CONFIG['MAX_TOKENS'],
            )
            return completion['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"An error occurred while generating the answer: {e}"