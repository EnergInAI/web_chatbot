import time
import google.generativeai as genai
from config import GEMINI_API_KEY, CHAT_MODEL

genai.configure(api_key=GEMINI_API_KEY)

# ===============================
# PER-IP RATE LIMITING
# ===============================

QUERY_LIMIT = 5               # free queries allowed per IP
RESET_TIME = 6 * 60 * 60      # 6 hours in seconds

# rate_state[ip] = { "count": X, "reset_at": timestamp }
rate_state = {}
rate_lock = False


def check_rate_limit(client_ip: str) -> bool:
    """Returns True if allowed, False if limit reached."""
    global rate_lock

    now = time.time()
    state = rate_state.get(client_ip)

    # First query for this IP
    if state is None:
        rate_state[client_ip] = {"count": 1, "reset_at": now + RESET_TIME}
        return True

    # Reset window if expired
    if now > state["reset_at"]:
        rate_state[client_ip] = {"count": 1, "reset_at": now + RESET_TIME}
        return True

    # Limit reached
    if state["count"] >= QUERY_LIMIT:
        return False

    # Allow + increment
    state["count"] += 1
    return True


# ===============================
# LLM FUNCTION
# ===============================

def generate_answer(query, context, client_ip="unknown"):

    # ==========================================
    #  GREETING HANDLER (added patch)
    # ==========================================
    greetings = {
        "hi", "hii", "hello", "hey", "hiya", "yo", "hola",
        "hi there", "hello there", "hey there",
        "good morning", "good afternoon", "good evening",
        "howdy", "greetings",
        "hi assistant", "hello assistant", "hey assistant"
    }

    q = query.lower().strip()

    # If the user input matches a greeting → return welcome message
    if q in greetings:
        return (
            "Hello Smart Consumer, Welcome to EnergInAI! "
            "How may I assist you today?"
        )

    # ==========================================
    # RATE LIMIT CHECK
    # ==========================================
    if not check_rate_limit(client_ip):
        return (
            "⚠️ You have reached your free query limit for now.<br><br>"
            "👉 Please book a free EnergInAI demo:<br>"
            "<a href='https://www.energinai.com/get-started' target='_blank' "
            "style='color:#1a56db; text-decoration:underline; font-weight:600;'>"
            "Click here to get started</a><br><br>"
            "Our team will reach out to you soon."
        )

    # ==========================================
    # NO CONTEXT (NOT FOUND IN RAG)
    # ==========================================
    if not context.strip():
        return (
            "👉 Please book a free EnergInAI demo to get assistance:<br>"
            "<a href='https://www.energinai.com/get-started' target='_blank' "
            "style='color:#1a56db; text-decoration:underline; font-weight:600;'>"
            "Click here to get started</a>"
        )

    # ==========================================
    # GENERATE ANSWER WITH CONTEXT
    # ==========================================
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
   "NOT_AVAILABLE"
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

    # ==========================================
    # RESPONSE NOT FOUND IN KNOWLEDGE BASE
    # ==========================================
    if "not_available" in text.lower():
        return (
            "👉 Please book a free EnergInAI demo to get assistance:<br>"
            "<a href='https://www.energinai.com/get-started' target='_blank' "
            "style='color:#1a56db; text-decoration:underline; font-weight:600;'>"
            "Click here to get started</a>"
        )

    # ==========================================
    # RETURN LEGIT MODEL ANSWER
    # ==========================================
    return text
