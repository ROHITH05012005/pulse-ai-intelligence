from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import uvicorn
import os

# Hugging Face Inference API Config
API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {}

app = FastAPI(title="Pulse AI: Emotion Classifier")

class TextRequest(BaseModel):
    text: str

import asyncio

@app.post("/classify")
async def classify_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Try up to 3 times if the model is loading
    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    API_URL, 
                    headers=headers, 
                    json={"inputs": request.text},
                    timeout=20.0
                )
                results = response.json()
            
            print(f"DEBUG: API output (Attempt {attempt+1}): {results}")

            # Handle "Model is loading" state
            if isinstance(results, dict) and "error" in results:
                error_msg = results.get("error", "Unknown error")
                if "loading" in error_msg.lower() and attempt < 2:
                    print("AI is still loading, waiting 5 seconds before retry...")
                    await asyncio.sleep(5)
                    continue
                raise Exception(error_msg)
            
            # Format results
            if isinstance(results, list) and len(results) > 0:
                data = results[0] if isinstance(results[0], list) else results
                formatted_results = sorted(data, key=lambda x: x['score'], reverse=True)
                
                return {
                    "text": request.text,
                    "classifications": formatted_results,
                    "top_emotion": formatted_results[0]['label'],
                    "status": "success"
                }
            
            raise Exception("Invalid AI response")

        except Exception as e:
            if attempt == 2: # Last attempt failed
                print(f"FINAL ERROR: {str(e)}")
                return {
                    "text": request.text,
                    "classifications": [],
                    "top_emotion": "neutral",
                    "status": "error",
                    "message": "The AI is deep in thought. Please try again in 30 seconds!"
                }
            await asyncio.sleep(2)

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
