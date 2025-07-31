import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-pro")

def generate_response(user_text: str, emotion: str):
    prompt = f"""
    Kullanıcı şu mesajı yazdı: "{user_text}"

    Bu mesajın duygu durumu: {emotion}

    Bu kişiye duygu durumuna uygun, anlayışlı ve nazik bir chatbot yanıtı üret.
    """
    response = model.generate_content(prompt)
    return response.text
