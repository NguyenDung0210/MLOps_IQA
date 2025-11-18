from fastapi import APIRouter
from fastapi.responses import HTMLResponse


root_router = APIRouter()


@root_router.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <body>
            <h1>Welcome to IQA API</h1>
        </body>
    </html>
    """
