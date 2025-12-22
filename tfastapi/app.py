# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from detector import detect_and_classify

app = FastAPI(title="L-Shape ML Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "OK", "message": "FastAPI ML Detection running."}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    result = detect_and_classify(contents)
    return {"detections": result}
