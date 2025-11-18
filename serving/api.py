from fastapi import FastAPI
from routers.health import health_router
from routers.predict import predict_router
from routers.root import root_router
import uvicorn


app = FastAPI()


app.include_router(health_router)
app.include_router(root_router)
app.include_router(predict_router)


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
