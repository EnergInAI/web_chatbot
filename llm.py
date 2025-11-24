import time
import google.generativeai as genai
from config import GEMINI_API_KEY, CHAT_MODEL

genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# PER-IP RATE LIMITING
# ===============================

QUERY_LIMIT = 5               # free queries allowed per IP
RESET_TIME = 6 * 60 * 60      # 6 hours in seconds

# store rate usage per client_ip:
#   rate_state[ip] = { "count": X, "reset_at": timestamp }
rate_state = {}
rate_lock = False  # primitive lock to avoid overwriting (sufficient for single-threaded Render)

def check_rate_limit(client_ip: str) -> bool:
    """Returns True if user is allowed, False if limit reached."""
    global rate_lock

    now = time.time()
    state = rate_state.get(client_ip)

    # First time this IP appears
    if state is None:
        rate_state[client_ip] = {"count": 1, "reset_at": now + RESET_TIME}
        return True

    # Reset the window if expired
    if now > state["reset_at"]:
        rate_state[client_ip] = {"count": 1, "reset_at": now + RESET_TIME}
        return True

    # Block if limit reached
    if state["count"] >= QUERY_LIMIT:
        return False

    # Otherwise allow & increment
    state["count"] += 1
    return True


# ===============================
# LLM FUNCTION
# ===============================

def generate_answer(query, context, client_ip="unknown"):

    # -------- RATE LIMIT CHECK --------
    if not check_rate_limit(client_ip):
        return (
            "⚠️ You have reached your free query limit for now.\n\n"
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7\n\n"
            "Our team will reach out to you soon."
        )

    # -------- RAG CONTEXT CHECK --------
    if not context.strip():
        return (
            "Not found in knowledge base.\n\n"
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7"
        )

    # -------- GENERATE ANSWER --------
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

    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        response = model.generate_content(prompt)
        text = response.text.strip()
    except Exception as e:
        print("Gemini ERROR:", e)
        return "Sorry, something went wrong while generating the response."

    if "not found" in text.lower():
        return (
            "Not found in knowledge base.\n\n"
            "👉 Please contact EnerginAI — Book a free demo:\n"
            "https://forms.gle/uAcYEV2r69HKPg3x7"
        )

    return text
