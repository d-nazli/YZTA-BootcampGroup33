import pickle

def load_model():
    with open("model/sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_sentiment(model, processed_text):
    # processed_text: string -> convert to list
    features = [processed_text]  # Örn. TF-IDF dönüşümü gerekiyorsa burada yapılmalı
    prediction = model.predict(features)[0]
    return prediction
