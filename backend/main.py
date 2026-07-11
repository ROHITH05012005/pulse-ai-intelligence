from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
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

# Hugging Face Inference API details
API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
# The token should be set in Render's environment variables as HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Pulse AI Backend is running with HF API!"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Call Hugging Face API
        response = requests.post(API_URL, headers=headers, json={"inputs": request.text})
        results = response.json()
        
        # Check if the API returned an error (e.g. model loading)
        if isinstance(results, dict) and "error" in results:
            raise HTTPException(status_code=503, detail=results["error"])
            
        # The API returns a list of lists: [[{"label": "joy", "score": 0.9}, ...]]
        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            classifications = results[0]
            formatted_results = sorted(classifications, key=lambda x: x['score'], reverse=True)
            
            return {
                "text": request.text,
                "classifications": formatted_results,
                "top_emotion": formatted_results[0]['label']
            }
        else:
            raise ValueError(f"Unexpected API response format: {results}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference API Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
