"""Приложение FastAPI: API прогноза отчисления и дашборд."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException

from src.api.endpoints import router, startup
from src.schemas.student import ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGE = Path(__file__).parent.parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app):
    try:
        startup()
    except Exception as exc:  # например, нет интернета для первой загрузки датасета
        logger.error("Модель не готова: %s", exc)
    yield


app = FastAPI(
    title="Риск отчисления студентов",
    description="Прогноз исхода учёбы — отчислят, ещё учится или выпустится — в три момента: "
                "при поступлении, после 1-го и после 2-го семестра.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/", include_in_schema=False)
async def page():
    return FileResponse(PAGE)


def error(code, message, details=""):
    return JSONResponse(status_code=code,
                        content=ErrorResponse(code=code, message=message, details=details).model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(exc.detail)
    return error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    details = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}" for e in exc.errors())
    logger.error(details)
    return error(422, "Неверные данные запроса", details)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(exc)
    return error(500, "Внутренняя ошибка сервера", str(exc))
