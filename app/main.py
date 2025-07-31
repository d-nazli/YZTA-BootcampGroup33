from fastapi import FastAPI
from app.schemas import SentimentRequest, ChatResponse
from app.model import load_model, predict_sentiment
from app.gemini import generate_response
from app.utils import preprocess_text

app = FastAPI()
model = load_model()


@app.post("/chat", response_model=ChatResponse)
def chat(request: SentimentRequest):
    processed = preprocess_text(request.text)
    emotion = predict_sentiment(model, processed)
    chatbot_reply = generate_response(request.text, emotion)
    return ChatResponse(
        sentiment=emotion,
        response=chatbot_reply
    )
