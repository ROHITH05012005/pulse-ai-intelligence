from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import os

app = FastAPI(title="Pulse AI: Emotion Classifier")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Emotion Classification Pipeline
print("Loading emotion classification model...")
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
    return {"message": "Pulse AI Backend is running!"}

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Perform classification
        # For top_k=None, results is a list of lists of dictionaries
        results = classifier(request.text)
        print(f"DEBUG: Classifier output: {results}")
        
        # results[0] is the list of all emotions for the first (and only) text input
        formatted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)
        
        return {
            "text": request.text,
            "classifications": formatted_results,
            "top_emotion": formatted_results[0]['label']
        }
    except Exception as e:
        print(f"ERROR during classification: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

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
