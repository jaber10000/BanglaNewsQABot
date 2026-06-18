import google.generativeai as genai
from config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("models/gemini-2.5-flash")

def generate_answer(context, question):

    prompt = f"""
তুমি একটি Bangla News Assistant।

নিচের context ব্যবহার করে প্রশ্নের উত্তর দাও।

Context:
{context}

Question:
{question}

উত্তর:
"""

    response = model.generate_content(prompt)

    return response.text