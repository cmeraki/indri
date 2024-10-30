# Inspired by https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/launcher.py

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response
from http import HTTPStatus

from vllm.engine.async_llm_engine import AsyncEngineDeadError
from vllm.engine.multiprocessing import MQEngineDeadError

from .logger import get_logger

logger = get_logger(__name__)

def _add_shutdown_handlers(app: FastAPI, server: uvicorn.Server) -> None:
    """Adds handlers for fatal errors that should crash the server"""

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, __):
        """On generic runtime error, check to see if the engine has died.
        It probably has, in which case the server will no longer be able to
        handle requests. Trigger a graceful shutdown with a SIGTERM."""

        engine = request.app.state.engine_client
        if (engine.errored and not engine.is_running):
            logger.critical("AsyncLLMEngine has failed, terminating server process")
            server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.exception_handler(AsyncEngineDeadError)
    async def async_engine_dead_handler(_, __):
        """Kill the server if the async engine is already dead. It will not handle any further requests."""
        logger.critical("AsyncLLMEngine is already dead, terminating server process")
        server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)

    @app.exception_handler(MQEngineDeadError)
    async def mq_engine_dead_handler(_, __):
        """Kill the server if the mq engine is already dead. It will
        not handle any further requests."""
        logger.critical("MQLLMEngine is already dead, terminating server process")
        server.should_exit = True

        return Response(status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
