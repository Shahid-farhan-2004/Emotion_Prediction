from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from keras.models import load_model
import numpy as np
import pickle
import re

model_path="Artifacts/BiGRU_model.keras"
tokenizer_path="Artifacts/tokenizer.pkl"
max_sequence_length=50
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}
def preprocess_text(text:str)->str:
    text=text.lower()
    text=re.sub(r"'","",text)
    text=re.sub(r"[^a-z0-9\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    return text

class textInput(BaseModel):
    text:str=Field(
        ...,
        min_length=1,
        max_length=2000,
        description="the sentence to analyze",
        json_schema_extra={"example":"i feel so happy and excited"}
    )

class predictionresponse(BaseModel):
    text:str
    predicted_emotion:str
    confidence:float
    all_probabilities:dict[str,float]

class healthresponse(BaseModel):
    status:str
    model_loaded:bool

dl_model={}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading the model and tokenizer")
    dl_model["model"]=load_model(model_path)
    with open(tokenizer_path,'rb') as file:
        dl_model["tokenizer"]=pickle.load(file)
    print("Model loaded successfully")
    yield
    dl_model.clear()

app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount('/static',StaticFiles(directory="static"),name="static")

@app.get("/",include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')

@app.get("/health",response_model=healthresponse)
def health_check():
    return healthresponse(status="the server is running",model_loaded=bool(dl_model))

@app.post("/predict",response_model=predictionresponse)
def predict_emotion(text_input:textInput):
    BiGRU_model=dl_model["model"]
    tokenizer_model=dl_model["tokenizer"]
    if BiGRU_model is None or tokenizer_model is None:
        raise HTTPException(status_code=503,detail="model is not loaded,try again later")
    cleaned_text=preprocess_text(text_input.text)
    tokenized_text=tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence=pad_sequences(
        tokenized_text,
        maxlen=max_sequence_length,
        padding="post",
        truncating='post'
    )

    probabilities=BiGRU_model.predict(padded_sequence)[0]

    top_emotion_index=int(np.argmax(probabilities))
    all_probabilitie={
        label:float(prob) for prob,label in zip(probabilities,emotion_labels)
    }
    return predictionresponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilitie
    )






















