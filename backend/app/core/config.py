"""Centralized Application Configuration for Argus.

Enforces Part A Zero-Hardcoding standard:
- Validated environment-driven settings with safe, documented defaults
- Centralized model parameters, timeouts, and heuristics
- Strict isolation of environment configs
"""

import json
from typing import List, Union, Optional
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    """Application settings loaded from environment variables and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Argus"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Local Ollama Inference Settings (Phase 3 & 3.1)
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_TIMEOUT_SECONDS: float = 60.0
    OLLAMA_STATUS_TIMEOUT_SECONDS: float = 4.0

    # Model Generation Hyperparameters
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_TOP_P: float = 0.9

    # Optimization & Token Estimation Heuristics
    CHARS_PER_TOKEN_HEURISTIC: float = 4.0
    MAX_VALIDATION_RETRY_ATTEMPTS: int = 1

    # Frontier OpenAI Provider Settings (Phase 4.1 & 4.2)
    ENABLE_UPSTREAM_DISPATCH: bool = False
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_MODEL: str = "gpt-6.1-sol"
    OPENAI_TIMEOUT_SECONDS: float = 60.0
    OPENAI_MAX_RETRIES: int = 2
    OPENAI_REASONING_EFFORT: Optional[str] = "medium"

    # Multi-Provider Integration Settings (Phase 7)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_DEFAULT_MODEL: str = "gemini-2.0-flash"
    GEMINI_TIMEOUT_SECONDS: float = 60.0

    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_DEFAULT_MODEL: str = "deepseek-chat"
    DEEPSEEK_TIMEOUT_SECONDS: float = 60.0

    OLLAMA_LLM_TIMEOUT_SECONDS: float = 60.0

    # Intelligent Model Routing Settings (Phase 6)
    ROUTING_DEFAULT_STRATEGY: str = "balanced"
    ROUTING_WEIGHT_QUALITY: float = 0.40
    ROUTING_WEIGHT_COST: float = 0.35
    ROUTING_WEIGHT_LATENCY: float = 0.25
    ROUTING_DEFAULT_MODEL: str = "gemini-2.0-flash"
    ROUTING_FALLBACK_CHAIN: List[str] = [
        "gemini-2.0-flash",
        "deepseek-chat",
        "qwen2.5:1.5b",
    ]

    # Cost and Telemetry Settings (Phase 8)
    TELEMETRY_ENABLED: bool = True
    TELEMETRY_STORAGE_BACKEND: str = "sqlite"
    TELEMETRY_DB_PATH: str = "argus_telemetry.db"
    TELEMETRY_MAX_MEMORY_RECORDS: int = 1000

    # TinyFish Browser Automation Settings (Phase 9)
    TINYFISH_ENABLED: bool = False
    TINYFISH_API_KEY: Optional[str] = None
    TINYFISH_BASE_URL: str = "https://api.tinyfish.io/v1"
    TINYFISH_TIMEOUT_SECONDS: float = 120.0
    TINYFISH_MAX_CONCURRENT_TASKS: int = 2

    # RABCO Phase 5 Research Settings (Counterfactual Risk Auditing & Selective Recovery)
    RABCO_MAX_RECOVERY_ATTEMPTS: int = 2
    """Maximum sequential recovery attempts in bounded restoration loop.
    Provisional baseline default: 2. This is an experimental starting value,
    not an empirically verified optimum.
    """

    RABCO_MAX_NON_MANDATORY_FAILURE_RATE: float = 0.0
    """Provisional tolerance threshold for non-mandatory probe failures (0.0 to 1.0).
    Default is 0.0 (provisional research baseline). Non-zero tolerance allows non-critical
    probes to fail without triggering bypass or recovery.
    Note: Mandatory evidence loss is a separate inviolable safety invariant and is never tolerated.
    """

    RABCO_ENABLE_SEMANTIC_AUDIT: bool = False
    """Whether independent semantic LLM auditing is enabled by default in RABCO Phase 5.
    Defaults to False (deterministic structural/provenance auditing is primary).
    """

    RABCO_AUDITOR_OLLAMA_MODEL: Optional[str] = None
    """Optional distinct model identifier for counterfactual semantic auditing.
    If None, falls back to OLLAMA_MODEL.
    RESEARCH NOTE: Sharing the default OLLAMA_MODEL runtime provides prompt-level
    independence only, NOT physical model independence. Physical model independence
    requires configuring a distinct secondary model endpoint or weight set.
    """

    RABCO_AUDITOR_OLLAMA_BASE_URL: Optional[str] = None
    """Optional distinct base URL for independent auditor model inference.
    If None, falls back to OLLAMA_BASE_URL.
    """

    RABCO_AUDITOR_TEMPERATURE: float = 0.0
    """Inference temperature for independent counterfactual auditor model (0.0 to 1.0).
    Provisional default: 0.0 for deterministic evaluation.
    """

    # RABCO Phase 5B: Counterfactual Probe Generation Settings
    # These settings govern the probe generator independently of the auditor.

    RABCO_PROBE_GENERATION_ENABLED: bool = True
    """Whether LLM-assisted probe generation is enabled in Phase 5B.
    When False, only deterministic template-based probes are produced.
    Independent of RABCO_ENABLE_SEMANTIC_AUDIT which controls the Phase 5C auditor.
    Provisional default: True (enabled).
    """

    RABCO_PROBE_GENERATION_MODEL: Optional[str] = None
    """Ollama model identifier for LLM-assisted counterfactual probe generation.
    If None, falls back to OLLAMA_MODEL.
    RESEARCH NOTE: Sharing the base OLLAMA_MODEL provides prompt-level independence only.
    Physical independence requires a distinct model endpoint or weight set.
    """

    RABCO_PROBE_GENERATION_BASE_URL: Optional[str] = None
    """Base URL for Ollama probe generation inference.
    If None, falls back to OLLAMA_BASE_URL.
    """

    RABCO_PROBE_GENERATION_TIMEOUT_SECONDS: float = 30.0
    """Timeout in seconds for Ollama probe generation requests.
    Provisional default: 30.0 seconds.
    """

    RABCO_PROBE_GENERATION_MAX_CANDIDATES: int = 5
    """Maximum number of LLM-generated probe candidates to request per generation call.
    Provisional default: 5. Candidates are subject to grounding validation and may be
    rejected if they reference non-existent evidence IDs or fail structural checks.
    """

    RABCO_PROBE_DEDUPLICATION_ENABLED: bool = True
    """Whether cross-probe question deduplication is applied during generation.
    Deduplication is case-insensitive and punctuation-normalized.
    Provisional default: True.
    """

    # RABCO Phase 5C: Risk Model Scoring Weights (Provisional Baseline)
    # Scientific Principle: These weights are heuristic research baselines, NOT calibrated probabilities.
    RABCO_RISK_WEIGHT_PROBE_FAILURE: float = 0.30
    RABCO_RISK_WEIGHT_MANDATORY_LOSS: float = 0.30
    RABCO_RISK_WEIGHT_DEPENDENCY_FAILURE: float = 0.15
    RABCO_RISK_WEIGHT_CONTRADICTION_FAILURE: float = 0.15
    RABCO_RISK_WEIGHT_SUMMARY_UNCERTAINTY: float = 0.10

    @field_validator(
        "ROUTING_WEIGHT_QUALITY",
        "ROUTING_WEIGHT_COST",
        "ROUTING_WEIGHT_LATENCY",
    )
    @classmethod
    def validate_routing_weight_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Routing weights must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_routing_weights_aggregate(self) -> "Settings":
        weights = (
            self.ROUTING_WEIGHT_QUALITY,
            self.ROUTING_WEIGHT_COST,
            self.ROUTING_WEIGHT_LATENCY,
        )
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Aggregate routing weights must sum to 1.0 within 0.01 tolerance, got sum={total:.4f}"
            )
        return self

    @field_validator(
        "RABCO_RISK_WEIGHT_PROBE_FAILURE",
        "RABCO_RISK_WEIGHT_MANDATORY_LOSS",
        "RABCO_RISK_WEIGHT_DEPENDENCY_FAILURE",
        "RABCO_RISK_WEIGHT_CONTRADICTION_FAILURE",
        "RABCO_RISK_WEIGHT_SUMMARY_UNCERTAINTY",
    )
    @classmethod
    def validate_risk_weight_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Risk indicator weights must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def validate_risk_weights_aggregate(self) -> "Settings":
        weights = (
            self.RABCO_RISK_WEIGHT_PROBE_FAILURE,
            self.RABCO_RISK_WEIGHT_MANDATORY_LOSS,
            self.RABCO_RISK_WEIGHT_DEPENDENCY_FAILURE,
            self.RABCO_RISK_WEIGHT_CONTRADICTION_FAILURE,
            self.RABCO_RISK_WEIGHT_SUMMARY_UNCERTAINTY,
        )
        total = sum(weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Aggregate RABCO risk weights must sum to 1.0 within 0.01 tolerance, got sum={total:.4f}"
            )
        return self

    @field_validator("RABCO_AUDITOR_TEMPERATURE")
    @classmethod
    def validate_auditor_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("RABCO_AUDITOR_TEMPERATURE must be between 0.0 and 1.0")
        return v

    @field_validator("RABCO_PROBE_GENERATION_TIMEOUT_SECONDS")
    @classmethod
    def validate_probe_generation_timeout(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("RABCO_PROBE_GENERATION_TIMEOUT_SECONDS must be positive")
        return v

    @field_validator("RABCO_PROBE_GENERATION_MAX_CANDIDATES")
    @classmethod
    def validate_probe_max_candidates(cls, v: int) -> int:
        if v < 1:
            raise ValueError("RABCO_PROBE_GENERATION_MAX_CANDIDATES must be at least 1")
        return v



    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


settings = Settings()
