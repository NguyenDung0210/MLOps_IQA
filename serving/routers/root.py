from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from serving.config import templates

root_router = APIRouter()

@root_router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
