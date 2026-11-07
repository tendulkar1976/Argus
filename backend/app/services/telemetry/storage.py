"""Persistence backends for the Cost & Telemetry Engine (Phase 8).

Implements:
- Abstract TelemetryStorage contract.
- SQLiteTelemetryStorage: Zero-dependency, thread-safe, indexed SQLite storage.
- InMemoryTelemetryStorage: In-memory circular buffer fallback.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from app.services.telemetry.models import TelemetryQueryFilter, TelemetryRecord, TelemetrySummary

logger = logging.getLogger(__name__)


class TelemetryStorage(ABC):
    """Abstract interface defining telemetry persistence operations."""

    @abstractmethod
    def save(self, record: TelemetryRecord) -> None:
        """Persist a telemetry record."""
        pass

    @abstractmethod
    def get_by_request_id(self, request_id: str) -> Optional[TelemetryRecord]:
        """Fetch a record by unique request ID."""
        pass

    @abstractmethod
    def query(self, filters: TelemetryQueryFilter) -> List[TelemetryRecord]:
        """Query records matching filter criteria."""
        pass

    @abstractmethod
    def get_summary(self, hours: Optional[int] = None) -> TelemetrySummary:
        """Calculate aggregated analytics over optional trailing hour window."""
        pass


class InMemoryTelemetryStorage(TelemetryStorage):
    """Circular buffer in-memory storage for test and ephemeral environments."""

    def __init__(self, max_records: int = 1000) -> None:
        self._max_records = max_records
        self._records: deque[TelemetryRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def save(self, record: TelemetryRecord) -> None:
        with self._lock:
            self._records.append(record)

    def get_by_request_id(self, request_id: str) -> Optional[TelemetryRecord]:
        with self._lock:
            for rec in reversed(self._records):
                if rec.request_id == request_id:
                    return rec
            return None

    def query(self, filters: TelemetryQueryFilter) -> List[TelemetryRecord]:
        with self._lock:
            matches: List[TelemetryRecord] = []
            for rec in reversed(self._records):
                if filters.provider and rec.provider != filters.provider:
                    continue
                if filters.model and rec.model != filters.model:
                    continue
                if filters.task_category and rec.task_category != filters.task_category:
                    continue
                if filters.status and rec.status != filters.status:
                    continue
                if filters.start_time and rec.timestamp < filters.start_time:
                    continue
                if filters.end_time and rec.timestamp > filters.end_time:
                    continue
                matches.append(rec)
            return matches[filters.offset : filters.offset + filters.limit]

    def get_summary(self, hours: Optional[int] = None) -> TelemetrySummary:
        with self._lock:
            now = datetime.now(timezone.utc)
            records = list(self._records)

        if hours is not None:
            cutoff = now.timestamp() - (hours * 3600)
            records = [r for r in records if r.timestamp.timestamp() >= cutoff]

        total_requests = len(records)
        if total_requests == 0:
            return TelemetrySummary(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                fallback_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_cached_tokens=0,
                total_tokens=0,
                total_estimated_cost_usd=0.0,
                average_latency_ms=0.0,
            )

        success = sum(1 for r in records if r.status == "success")
        failed = sum(1 for r in records if r.status == "error")
        fallback = sum(1 for r in records if r.fallback_used)
        input_tokens = sum(r.input_tokens or 0 for r in records)
        output_tokens = sum(r.output_tokens or 0 for r in records)
        cached_tokens = sum(r.cached_tokens or 0 for r in records)
        total_tokens = sum(r.total_tokens or 0 for r in records)
        cost = sum(r.estimated_cost_usd or 0.0 for r in records)
        avg_latency = sum(r.latency_ms for r in records) / total_requests

        by_provider: Dict[str, int] = {}
        by_model: Dict[str, int] = {}
        by_task: Dict[str, int] = {}
        cost_by_provider: Dict[str, float] = {}

        for r in records:
            by_provider[r.provider] = by_provider.get(r.provider, 0) + 1
            by_model[r.model] = by_model.get(r.model, 0) + 1
            by_task[r.task_category] = by_task.get(r.task_category, 0) + 1
            if r.estimated_cost_usd:
                cost_by_provider[r.provider] = round(
                    cost_by_provider.get(r.provider, 0.0) + r.estimated_cost_usd, 6
                )

        return TelemetrySummary(
            total_requests=total_requests,
            successful_requests=success,
            failed_requests=failed,
            fallback_requests=fallback,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            total_estimated_cost_usd=round(cost, 6),
            average_latency_ms=round(avg_latency, 2),
            requests_by_provider=by_provider,
            requests_by_model=by_model,
            requests_by_task=by_task,
            cost_by_provider=cost_by_provider,
        )


class SQLiteTelemetryStorage(TelemetryStorage):
    """ACID-compliant SQLite telemetry storage with connection-per-thread management."""

    def __init__(self, db_path: str = "argus_telemetry.db") -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_records (
                    request_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_category TEXT NOT NULL,
                    routing_decision TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cached_tokens INTEGER,
                    total_tokens INTEGER,
                    estimated_cost_usd REAL,
                    provider_reported_cost_usd REAL,
                    cost_status TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    time_to_first_token_ms INTEGER,
                    status TEXT NOT NULL,
                    error_category TEXT,
                    retry_count INTEGER NOT NULL,
                    fallback_used INTEGER NOT NULL,
                    fallback_model_from TEXT,
                    rabco_processing_time_ms INTEGER,
                    prompt_optimization_metrics TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry_records (timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_provider_model ON telemetry_records (provider, model)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_status ON telemetry_records (status)"
            )

    def save(self, record: TelemetryRecord) -> None:
        conn = self._get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO telemetry_records VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    record.request_id,
                    record.timestamp.isoformat(),
                    record.provider,
                    record.model,
                    record.task_category,
                    json.dumps(record.routing_decision) if record.routing_decision else None,
                    record.input_tokens,
                    record.output_tokens,
                    record.cached_tokens,
                    record.total_tokens,
                    record.estimated_cost_usd,
                    record.provider_reported_cost_usd,
                    record.cost_status,
                    record.currency,
                    record.latency_ms,
                    record.time_to_first_token_ms,
                    record.status,
                    record.error_category,
                    record.retry_count,
                    1 if record.fallback_used else 0,
                    record.fallback_model_from,
                    record.rabco_processing_time_ms,
                    json.dumps(record.prompt_optimization_metrics)
                    if record.prompt_optimization_metrics
                    else None,
                ),
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TelemetryRecord:
        return TelemetryRecord(
            request_id=row["request_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            provider=row["provider"],
            model=row["model"],
            task_category=row["task_category"],
            routing_decision=json.loads(row["routing_decision"]) if row["routing_decision"] else None,
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cached_tokens=row["cached_tokens"],
            total_tokens=row["total_tokens"],
            estimated_cost_usd=row["estimated_cost_usd"],
            provider_reported_cost_usd=row["provider_reported_cost_usd"],
            cost_status=row["cost_status"],
            currency=row["currency"],
            latency_ms=row["latency_ms"],
            time_to_first_token_ms=row["time_to_first_token_ms"],
            status=row["status"],
            error_category=row["error_category"],
            retry_count=row["retry_count"],
            fallback_used=bool(row["fallback_used"]),
            fallback_model_from=row["fallback_model_from"],
            rabco_processing_time_ms=row["rabco_processing_time_ms"],
            prompt_optimization_metrics=json.loads(row["prompt_optimization_metrics"])
            if row["prompt_optimization_metrics"]
            else None,
        )

    def get_by_request_id(self, request_id: str) -> Optional[TelemetryRecord]:
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM telemetry_records WHERE request_id = ?", (request_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def query(self, filters: TelemetryQueryFilter) -> List[TelemetryRecord]:
        query_sql = "SELECT * FROM telemetry_records WHERE 1=1"
        params: List[Any] = []

        if filters.provider:
            query_sql += " AND provider = ?"
            params.append(filters.provider)
        if filters.model:
            query_sql += " AND model = ?"
            params.append(filters.model)
        if filters.task_category:
            query_sql += " AND task_category = ?"
            params.append(filters.task_category)
        if filters.status:
            query_sql += " AND status = ?"
            params.append(filters.status)
        if filters.start_time:
            query_sql += " AND timestamp >= ?"
            params.append(filters.start_time.isoformat())
        if filters.end_time:
            query_sql += " AND timestamp <= ?"
            params.append(filters.end_time.isoformat())

        query_sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([filters.limit, filters.offset])

        conn = self._get_connection()
        cursor = conn.execute(query_sql, params)
        return [self._row_to_record(r) for r in cursor.fetchall()]

    def get_summary(self, hours: Optional[int] = None) -> TelemetrySummary:
        conn = self._get_connection()
        where_clause = ""
        params: List[Any] = []
        if hours is not None:
            cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
            cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
            where_clause = "WHERE timestamp >= ?"
            params.append(cutoff_iso)

        cursor = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as failed_requests,
                SUM(CASE WHEN fallback_used = 1 THEN 1 ELSE 0 END) as fallback_requests,
                COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                COALESCE(SUM(cached_tokens), 0) as total_cached_tokens,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(estimated_cost_usd), 0.0) as total_estimated_cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as average_latency_ms
            FROM telemetry_records {where_clause}
            """,
            params,
        )
        row = cursor.fetchone()

        # Breakdown by provider
        p_cursor = conn.execute(
            f"SELECT provider, COUNT(*) as cnt, COALESCE(SUM(estimated_cost_usd), 0.0) as cost FROM telemetry_records {where_clause} GROUP BY provider",
            params,
        )
        by_provider = {}
        cost_by_provider = {}
        for r in p_cursor.fetchall():
            by_provider[r["provider"]] = r["cnt"]
            cost_by_provider[r["provider"]] = round(r["cost"], 6)

        # Breakdown by model
        m_cursor = conn.execute(
            f"SELECT model, COUNT(*) as cnt FROM telemetry_records {where_clause} GROUP BY model",
            params,
        )
        by_model = {r["model"]: r["cnt"] for r in m_cursor.fetchall()}

        # Breakdown by task
        t_cursor = conn.execute(
            f"SELECT task_category, COUNT(*) as cnt FROM telemetry_records {where_clause} GROUP BY task_category",
            params,
        )
        by_task = {r["task_category"]: r["cnt"] for r in t_cursor.fetchall()}

        return TelemetrySummary(
            total_requests=row["total_requests"] or 0,
            successful_requests=row["successful_requests"] or 0,
            failed_requests=row["failed_requests"] or 0,
            fallback_requests=row["fallback_requests"] or 0,
            total_input_tokens=row["total_input_tokens"] or 0,
            total_output_tokens=row["total_output_tokens"] or 0,
            total_cached_tokens=row["total_cached_tokens"] or 0,
            total_tokens=row["total_tokens"] or 0,
            total_estimated_cost_usd=round(row["total_estimated_cost_usd"] or 0.0, 6),
            average_latency_ms=round(row["average_latency_ms"] or 0.0, 2),
            requests_by_provider=by_provider,
            requests_by_model=by_model,
            requests_by_task=by_task,
            cost_by_provider=cost_by_provider,
        )
