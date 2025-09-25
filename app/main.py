# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI(title="Simple Text Generator (Bigram)")

# Tiny sample training corpus
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "It tells the story of Edmond Dantes who seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI with UV!"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Body example:
    {
      "start_word": "the",
      "length": 12
    }
    """
    text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": text}
