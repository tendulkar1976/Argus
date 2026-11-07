"""Centralized API Router for Argus."""

from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.optimizer import router as optimizer_router
from app.api.gateway import router as gateway_router
from app.api.execution import router as execution_router
from app.api.automation import router as automation_router

api_router = APIRouter()

# Include health router
api_router.include_router(health_router)

# Include optimizer router
api_router.include_router(optimizer_router)

# Include gateway dispatch router (Phase 4.2)
api_router.include_router(gateway_router)

# Include unified execution & telemetry router (Phase 10)
api_router.include_router(execution_router)

# Include browser automation router (Phase 9 & 10)
api_router.include_router(automation_router)

