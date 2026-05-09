from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import torch
import uvicorn
import os
import gc

# MEMORY OPTIMIZATION: Use a lighter model and limit CPU threads
torch.set_num_threads(1)

app = FastAPI(title="Pulse AI: Emotion Classifier")

# Initialize the Emotion Classification Pipeline LOCALLY
# We use a distilled model to stay under 512MB RAM
print("Loading lightweight emotion model...")
classifier = pipeline(
    "text-classification", 
    model="bhadresh-savani/distilbert-base-uncased-emotion", 
    top_k=None,
    device=-1 # Ensure CPU only
)
print("Model loaded successfully!")

class TextRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Perform local classification
        results = classifier(request.text)
        
        # Format results
        data = results[0]
        formatted_results = sorted(data, key=lambda x: x['score'], reverse=True)
        
        # Clear memory after heavy lifting
        gc.collect()
        
        return {
            "text": request.text,
            "classifications": formatted_results,
            "top_emotion": formatted_results[0]['label'],
            "status": "success"
        }

    except Exception as e:
        print(f"ERROR during classification: {str(e)}")
        return {
            "text": request.text,
            "classifications": [],
            "top_emotion": "neutral",
            "status": "error",
            "message": "AI is momentarily overwhelmed. Try a shorter sentence!"
        }

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
