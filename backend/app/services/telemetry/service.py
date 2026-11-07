"""Telemetry Service coordinating cost computation, metric logging, and persistence (Phase 8).

Guarantees:
- Safe telemetry recording: Storage errors are trapped and logged without failing model responses.
- Accurate decimal cost resolution using verified rates from ModelPricingRegistry.
- Standardized analytics queries and aggregations.
"""

import logging
from typing import Any, Dict, List, Optional
from app.core.config import settings
from app.services.cost_calculator import CostCalculator
from app.services.telemetry.models import TelemetryQueryFilter, TelemetryRecord, TelemetrySummary
from app.services.telemetry.storage import (
    InMemoryTelemetryStorage,
    SQLiteTelemetryStorage,
    TelemetryStorage,
)

logger = logging.getLogger(__name__)


class TelemetryService:
    """Service providing unified telemetry logging, cost calculation, and persistence."""

    _instance: Optional["TelemetryService"] = None

    def __init__(self, storage: Optional[TelemetryStorage] = None) -> None:
        if storage is not None:
            self._storage = storage
        else:
            backend = getattr(settings, "TELEMETRY_STORAGE_BACKEND", "sqlite").lower()
            if backend == "memory":
                max_recs = getattr(settings, "TELEMETRY_MAX_MEMORY_RECORDS", 1000)
                self._storage = InMemoryTelemetryStorage(max_records=max_recs)
            else:
                db_path = getattr(settings, "TELEMETRY_DB_PATH", "argus_telemetry.db")
                self._storage = SQLiteTelemetryStorage(db_path=db_path)

    @classmethod
    def get_instance(cls) -> "TelemetryService":
        """Get or initialize process singleton TelemetryService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_execution(
        self,
        request_id: str,
        provider: str,
        model: str,
        latency_ms: int,
        status: str,
        task_category: str = "general",
        routing_decision: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        provider_reported_cost_usd: Optional[float] = None,
        error_category: Optional[str] = None,
        retry_count: int = 0,
        fallback_used: bool = False,
        fallback_model_from: Optional[str] = None,
        rabco_processing_time_ms: Optional[int] = None,
        prompt_optimization_metrics: Optional[Dict[str, Any]] = None,
        time_to_first_token_ms: Optional[int] = None,
    ) -> Optional[TelemetryRecord]:
        """Compute costs and persist an audit telemetry record.

        Failure Policy:
        Telemetry persistence exceptions are trapped and logged as warnings.
        They do NOT propagate to corrupt client responses.
        """
        if not getattr(settings, "TELEMETRY_ENABLED", True):
            return None

        # Compute costs
        calculated_cost: Optional[float] = None
        cost_status: str

        if provider.lower() == "ollama":
            # Local model has zero external API cost
            calculated_cost = 0.0
            cost_status = "calculated_local"
        else:
            cost_res = CostCalculator.calculate_cost(
                model_id=model,
                actual_input_tokens=input_tokens,
                actual_output_tokens=output_tokens,
                cached_tokens=cached_tokens or 0,
                reasoning_tokens=reasoning_tokens or 0,
            )
            calculated_cost = cost_res.actual_cost_usd
            cost_status = cost_res.pricing_status

        record = TelemetryRecord(
            request_id=request_id,
            provider=provider,
            model=model,
            task_category=task_category,
            routing_decision=routing_decision,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=calculated_cost,
            provider_reported_cost_usd=provider_reported_cost_usd,
            cost_status=cost_status,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            status=status,
            error_category=error_category,
            retry_count=retry_count,
            fallback_used=fallback_used,
            fallback_model_from=fallback_model_from,
            rabco_processing_time_ms=rabco_processing_time_ms,
            prompt_optimization_metrics=prompt_optimization_metrics,
        )

        try:
            self._storage.save(record)
            return record
        except Exception as exc:
            logger.warning(
                "Failed to persist telemetry record for request_id=%s: %s",
                request_id,
                str(exc),
            )
            return record

    def get_by_request_id(self, request_id: str) -> Optional[TelemetryRecord]:
        return self._storage.get_by_request_id(request_id)

    def query(self, filters: TelemetryQueryFilter) -> List[TelemetryRecord]:
        return self._storage.query(filters)

    def get_summary(self, hours: Optional[int] = None) -> TelemetrySummary:
        return self._storage.get_summary(hours)
