from fastapi import FastAPI

app = FastAPI(
    title="AI Video Translation and Dubbing System"
)

@app.get("/")
def home():
    return {
        "status": "running",
        "project": "AI Video Translation and Dubbing"
    }