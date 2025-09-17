from fastapi import FastAPI

app = FastAPI(title="Sentiment Analysis API")

@app.get("/health")
def health():
    return {"status": "ok"}
