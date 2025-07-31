from pydantic import BaseModel

class SentimentRequest(BaseModel):
    text: str

class ChatResponse(BaseModel):
    sentiment: str
    response: str
