import google.generativeai as genai
import numpy as np
from config import GEMINI_API_KEY, EMBED_MODEL

genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text):
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(result["embedding"])
