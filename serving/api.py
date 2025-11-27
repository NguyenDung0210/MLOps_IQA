from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving.routers.health import health_router
from serving.routers.root import root_router
from serving.routers.predict import predict_router, ui_router

import uvicorn

app = FastAPI()

app.include_router(health_router)
app.include_router(root_router)
app.include_router(predict_router)
app.include_router(ui_router)

app.mount("/static", StaticFiles(directory="serving/static"), name="static")

if __name__ == "__main__":
    uvicorn.run("serving.api:app", host="0.0.0.0", port=8000, reload=True)
