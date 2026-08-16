from fastapi import FastAPI
from deployment.fastapi.routes.predict import router

app = FastAPI(
    title="Graph Fraud AI API",
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status":"healthy"}
