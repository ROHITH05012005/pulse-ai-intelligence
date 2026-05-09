from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import uvicorn
import os

# Hugging Face Inference API Config
API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {}

app = FastAPI(title="Pulse AI: Emotion Classifier")

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Call Hugging Face Inference API instead of local model
        response = requests.post(API_URL, headers=headers, json={"inputs": request.text})
        results = response.json()
        
        print(f"DEBUG: API output: {results}")
        
        # Format results (The API returns a list of dictionaries)
        # Handle cases where the API returns a list of lists
        if isinstance(results[0], list):
            formatted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        else:
            formatted_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        return {
            "text": request.text,
            "classifications": formatted_results,
            "top_emotion": formatted_results[0]['label']
        }
    except Exception as e:
        print(f"ERROR during classification: {str(e)}")
        raise HTTPException(status_code=500, detail="The AI Brain is taking a nap. Please try again in a moment!")

# --- PRODUCTION STATIC FILE SERVING ---
# Mount the built frontend files
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="./static"), name="static")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Serve the index.html for any route not starting with /api or /classify
        if full_path.startswith("classify") or full_path.startswith("api"):
            raise HTTPException(status_code=404)
        
        file_path = os.path.join("./static", full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse("./static/index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
