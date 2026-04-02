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
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    thread_id: str
    summary: str


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    result = await notebook_lm.ask(request.query, thread_id=request.thread_id)
    return AskResponse(answer=result.answer, thread_id=result.thread_id, summary=result.summary)
