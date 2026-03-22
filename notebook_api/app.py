from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import notebook_lm


@asynccontextmanager
async def lifespan(app: FastAPI):
    await notebook_lm.startup()
    yield


app = FastAPI(title="NotebookLM API", lifespan=lifespan)


@app.get("/", include_in_schema=False)
@app.get("/swagger", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    answer = await notebook_lm.ask(request.query)
    return AskResponse(answer=answer)
