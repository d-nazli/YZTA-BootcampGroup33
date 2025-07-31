from fastapi import FastAPI
from backend.schemas import SentimentRequest, SentimentResponse
from backend.model import load_model, predict_sentiment
from backend.utils import preprocess_text

app = FastAPI()
model = load_model()

@app.get("/")
def root():
    return {"message": "Duygu Durumu API'si çalışıyor."}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    processed = preprocess_text(request.text)
    prediction = predict_sentiment(model, processed)
    return SentimentResponse(sentiment=prediction)
