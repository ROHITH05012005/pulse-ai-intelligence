from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os

app = FastAPI(title="Pulse AI: Emotion Classifier")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Emotion Classification Pipeline
print("Loading emotion classification model (j-hartmann/emotion-english-distilroberta-base)...")
classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    top_k=None
)
print("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Pulse AI Local Backend is running!"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Perform classification
        results = classifier(request.text)
        
        # Format results for the frontend
        formatted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        
        return {
            "text": request.text,
            "classifications": formatted_results,
            "top_emotion": formatted_results[0]['label']
        }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
