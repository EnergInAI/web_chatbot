import google.generativeai as genai
from config import GEMINI_API_KEY, CHAT_MODEL

genai.configure(api_key=GEMINI_API_KEY)

def generate_answer(query, context):
    # If context blank → no answer found
    if not context.strip():
        return "Not found in knowledge base."

    prompt = f"""
You are a strict RAG chatbot.
Use ONLY the context provided below to answer the question.

--- CONTEXT ---
{context}

--- QUESTION ---
{query}

RULES:
1. Do NOT use outside knowledge.
2. Do NOT guess anything.
3. If the answer is not present in the context, reply EXACTLY:
   "Not found in knowledge base."
4. Only use English language.

Your answer:
"""

    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)

    # Extra protection – if LLM tries to hallucinate
    text = response.text.strip()
    if "not found" in text.lower():
        return "Not found in knowledge base."

    if query.lower() not in context.lower() and len(context.strip()) == 0:
        return "Not found in knowledge base."

    return text
