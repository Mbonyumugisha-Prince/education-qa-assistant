import asyncio
import json
import logging
import queue
import threading
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool

from streaming_assistant import StreamingEducationAssistant


logger = logging.getLogger("education_api")

Subject = Literal[
    "mathematics",
    "science",
    "python",
    "java",
    "technology",
    "history",
    "geography",
]

ResponseType = Literal[
    "greeting",
    "generated",
    "textbook_excerpts",
]


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: Subject
    question: str = Field(min_length=1, max_length=12000)

    # Optional field retained for compatibility.
    passage: str = Field(default="", max_length=20000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Enter a question.")

        return value

    @field_validator("passage")
    @classmethod
    def clean_passage(cls, value: str) -> str:
        return value.strip()


class AnswerResponse(BaseModel):
    request_id: str
    subject: Subject
    answer: str
    response_type: ResponseType
    duration_seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.answer_lock = threading.Lock()

    app.state.assistant = await run_in_threadpool(
        StreamingEducationAssistant
    )
    app.state.ready = True

    try:
        yield
    finally:
        app.state.ready = False

        # Wait for any active model operation before clearing it.
        await run_in_threadpool(app.state.answer_lock.acquire)

        try:
            app.state.assistant = None
        finally:
            app.state.answer_lock.release()


app = FastAPI(
    title="Education Assistant API",
    description=(
        "Educational questions across seven subjects. "
        "History returns selected textbook excerpts. "
        "Generated answers can contain errors."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:2000",
        "http://127.0.0.1:2000",
        "http://localhost:2020",
        "http://127.0.0.1:2020",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def acquire_assistant(request: Request):
    state = request.app.state

    if not getattr(state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="The assistant is not ready.",
        )

    if not state.answer_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail=(
                "The assistant is answering another question. "
                "Please try again shortly."
            ),
            headers={"Retry-After": "5"},
        )

    return state


def get_response_type(assistant, payload: AnswerRequest) -> ResponseType:
    if assistant.greeting_response(payload.question) is not None:
        return "greeting"

    if payload.subject == "history":
        return "textbook_excerpts"

    return "generated"


@app.get("/")
def root():
    return {
        "name": "Education Assistant API",
        "docs": "/docs",
        "health": "/health",
        "subjects": "/subjects",
        "answer": "/answer",
        "stream": "/answer/stream",
    }


@app.get("/health")
def health(request: Request):
    state = request.app.state

    if not getattr(state, "ready", False):
        raise HTTPException(
            status_code=503,
            detail="The assistant is not ready.",
        )

    return {
        "status": "ready",
        "model": "V6 step-100",
        "busy": state.answer_lock.locked(),
    }


@app.get("/subjects")
def subjects():
    labels = [
        ("mathematics", "Mathematics"),
        ("science", "Science"),
        ("python", "Python"),
        ("java", "Java"),
        ("technology", "Technology"),
        ("history", "History — textbook search"),
        ("geography", "Geography"),
    ]

    return {
        "subjects": [
            {
                "id": subject_id,
                "label": label,
                "requires_passage": False,
            }
            for subject_id, label in labels
        ]
    }


@app.post("/answer", response_model=AnswerResponse)
def answer(payload: AnswerRequest, request: Request):
    state = acquire_assistant(request)
    request_id = str(uuid4())
    started = perf_counter()

    try:
        assistant = state.assistant
        response_type = get_response_type(assistant, payload)

        result = assistant.answer(
            subject=payload.subject,
            question=payload.question,
            passage=payload.passage,
        )

        return AnswerResponse(
            request_id=request_id,
            subject=payload.subject,
            answer=result,
            response_type=response_type,
            duration_seconds=round(
                perf_counter() - started,
                3,
            ),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Answer request failed: %s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The assistant could not complete this request. "
                f"Reference: {request_id}"
            ),
        ) from error

    finally:
        state.answer_lock.release()


@app.post("/answer/stream")
async def answer_stream(payload: AnswerRequest, request: Request):
    state = acquire_assistant(request)

    request_id = str(uuid4())
    events = queue.Queue()
    stop_event = threading.Event()

    def worker():
        started = perf_counter()

        try:
            assistant = state.assistant
            response_type = get_response_type(assistant, payload)

            events.put({
                "type": "start",
                "request_id": request_id,
                "response_type": response_type,
            })

            emitted = False

            def emit(text):
                nonlocal emitted

                if text and not stop_event.is_set():
                    emitted = True

                    events.put({
                        "type": "delta",
                        "text": text,
                    })

            if stop_event.is_set():
                return

            result = assistant.answer_stream(
                subject=payload.subject,
                question=payload.question,
                passage=payload.passage,
                callback=emit,
                stop_event=stop_event,
            )

            if not isinstance(result, str):
                raise RuntimeError(
                    "The assistant returned an invalid answer."
                )

            if stop_event.is_set():
                return

            # Greetings and history search do not generate token streams.
            if not emitted:
                emit(result)

            events.put({
                "type": "done",
                "request_id": request_id,
                "subject": payload.subject,
                "answer": result,
                "response_type": response_type,
                "duration_seconds": round(
                    perf_counter() - started,
                    3,
                ),
            })

        except Exception as error:
            logger.exception(
                "Streaming request failed: %s",
                request_id,
            )

            if not stop_event.is_set():
                events.put({
                    "type": "error",
                    "detail": (
                        str(error)
                        if isinstance(error, ValueError)
                        else (
                            "The assistant could not finish the response. "
                            f"Reference: {request_id}"
                        )
                    ),
                })

        finally:
            # Only release when the worker actually finishes.
            state.answer_lock.release()

            # Internal queue signal; never sent to the browser.
            events.put(None)

    thread = threading.Thread(
        target=worker,
        name=f"answer-{request_id}",
        daemon=True,
    )

    try:
        thread.start()
    except Exception:
        state.answer_lock.release()
        raise

    async def event_stream():
        last_heartbeat = perf_counter()

        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    event = events.get_nowait()
                except queue.Empty:
                    if perf_counter() - last_heartbeat >= 10:
                        yield json.dumps({"type": "ping"}) + "\n"
                        last_heartbeat = perf_counter()

                    await asyncio.sleep(0.05)
                    continue

                # Check before JSON encoding to avoid sending null.
                if event is None:
                    break

                yield json.dumps(
                    event,
                    ensure_ascii=False,
                ) + "\n"

        finally:
            # The model checks this between generation steps.
            # History retrieval may need to finish its current operation.
            stop_event.set()

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )