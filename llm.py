import time
import google.generativeai as genai
from config import GEMINI_API_KEY, CHAT_MODEL

genai.configure(api_key=GEMINI_API_KEY)

# =========================================
# RATE LIMIT SYSTEM (5 queries per 6 hours)
# =========================================

QUERY_LIMIT = 5
RESET_TIME = 6 * 60 * 60   # 6 hours

user_state = {
    "count": 0,
    "reset_at": time.time() + RESET_TIME
}

def check_rate_limit():
    current_time = time.time()

    # Reset if time passed
    if current_time > user_state["reset_at"]:
        user_state["count"] = 0
        user_state["reset_at"] = current_time + RESET_TIME

    # If limit reached
    if user_state["count"] >= QUERY_LIMIT:
        return False

    # Allow usage
    user_state["count"] += 1
    return True


# =========================================
# MAIN FUNCTION
# =========================================

def generate_answer(query, context):

    # -------- RATE LIMIT CHECK --------
    if not check_rate_limit():
        return (
            "⚠️ You have reached your free query limit.For more queries\n\n"
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7\n\n"
            "Once you fill the form, our team will contact you."
        )

    # -------- NORMAL RAG LOGIC --------
    if not context.strip():
        return ("⚠️ You have reached your free query limit.For more queries\n\n"
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7\n\n"
            "Once you fill the form, our team will contact you.")

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
    text = response.text.strip()

    if "not found" in text.lower():
        return (
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7\n\n"
            "Once you fill the form, our team will contact you.")

    return text
