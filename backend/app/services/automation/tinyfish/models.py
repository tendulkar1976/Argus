"""Data contracts and schemas for TinyFish Browser Automation (Phase 9).

Separates browser automation workloads strictly from LLM text generation.
Enforces:
- Explicit task requests with strict parameter validation.
- Standardized execution statuses and normalized result structures.
- Sensitive credential protection.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TinyFishActionType(str, Enum):
    """Categorical actions supported by the browser automation boundary."""

    NAVIGATE = "navigate"
    EXTRACT = "extract"
    CLICK = "click"
    SCREENSHOT = "screenshot"
    EVALUATE = "evaluate"
    AUTOMATE_WORKFLOW = "automate_workflow"


class TinyFishTaskStatus(str, Enum):
    """Execution lifecycle status for browser automation tasks."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class TinyFishTaskRequest(BaseModel):
    """Validated client request to execute a bounded browser automation task."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(description="Unique identifier for the automation task")
    action: TinyFishActionType = Field(description="Browser automation action to execute")
    url: str = Field(min_length=8, description="Target webpage URL (must be valid http or https)")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (e.g. selectors, input text, extraction schema)",
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=300.0,
        description="Execution timeout in seconds",
    )
    capture_screenshot: bool = Field(
        default=False,
        description="Whether to capture and return a base64 encoded screenshot of the page state",
    )

    @field_validator("url")
    @classmethod
    def validate_url_protocol(cls, v: str) -> str:
        trimmed = v.strip()
        if not (trimmed.startswith("http://") or trimmed.startswith("https://")):
            raise ValueError("Target URL must begin with 'http://' or 'https://'")
        return trimmed


class TinyFishTaskResponse(BaseModel):
    """Normalized result returned from browser automation execution."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="Unique task identifier")
    action: TinyFishActionType = Field(description="Executed automation action")
    url: str = Field(description="Target URL")
    status: TinyFishTaskStatus = Field(description="Final execution outcome status")
    result_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted or computed result payload from the browser",
    )
    screenshot_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data if screenshot was requested and captured",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Detailed error message if automation execution failed",
    )
    execution_time_ms: int = Field(ge=0, description="Task execution duration in milliseconds")
    integration_mode: str = Field(
        description="Mode of execution: 'live_api', 'boundary_validated', or 'unconfigured'",
    )
