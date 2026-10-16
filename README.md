# Argus 🛡️

> **Intelligent Prompt Optimization Middleware**  
> Dramatically reduce main-LLM input-token costs with local model prompt compression while preserving intent and semantic constraints.

[![CI Pipeline](https://github.com/argus-ai/argus/actions/workflows/ci.yml/badge.svg)](https://github.com/argus-ai/argus/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React: 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Tailwind CSS: v4](https://img.shields.io/badge/Tailwind-v4-06B6D4.svg)](https://tailwindcss.com/)

---

## 📖 Table of Contents
- [Project Purpose & Scope](#project-purpose--scope)
- [Architecture & Boundaries](#architecture--boundaries)
- [Technology Stack](#technology-stack)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation & Local Run Instructions](#installation--local-run-instructions)
- [Testing & Verification](#testing--verification)
- [Planned Implementation Phases](#planned-implementation-phases)
- [Contributing & Development Guide](#contributing--development-guide)
- [License](#license)

---

## Project Purpose & Scope

Large Language Models (LLMs) incur significant costs scaled directly by input-token volume. Complex system prompts, verbose instructions, repeated examples, and boilerplate constraints quickly inflate API bills on premium frontier models.

**Argus** functions as an intelligent middleware gateway:
1. **Intercepts** client prompts before transmission to expensive frontier models.
2. **Extracts & Locks** vital semantic constraints, task parameters, and output formats.
3. **Compresses & Optimizes** prompt wording locally using a lightweight local LLM (**Qwen2.5-1.5B-Instruct** running via **Ollama**).
4. **Verifies Constraint Integrity** to ensure zero loss of intent or answer quality.
5. **Dispatches** the optimized prompt to the target main LLM (OpenAI, Anthropic, Gemini, etc.), returning both the answer and transparent token/cost savings telemetry.

---

## Architecture & Boundaries

Argus enforces strict architectural boundaries:

```
[ Frontend: React + Vite + Tailwind ]
            │  (REST API calls via typed contracts)
            ▼
[ Backend API: FastAPI Router ]
            │  (Schema validation via Pydantic)
            ▼
[ Services Layer: Prompt Compression & Orchestration ]
       ├──▶ [ Local Inference Tier: Ollama (Qwen2.5-1.5B) ]
       └──▶ [ Frontier LLM Tier: OpenAI / Anthropic / Gemini ]
```

- **Frontend Isolation**: Frontend components serve purely as presentation and configuration interfaces. They contain **no** backend business logic, prompt tokenization algorithms, or direct provider API keys.
- **Backend Service Layer**: Handles orchestration, compression heuristics, and provider dispatch.
- **Security**: Provider API keys remain strictly on the backend. Model weights and environments are excluded from source control.

For in-depth architecture diagrams and specifications, see [`docs/architecture.md`](docs/architecture.md).

---

## Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, TypeScript, Vite, Tailwind CSS v4, Lucide Icons, Vitest |
| **Backend** | Python 3.10–3.13, FastAPI, Pydantic v2, Pydantic-Settings, Uvicorn, Pytest |
| **Local Inference** | Ollama with `Qwen2.5-1.5B-Instruct` (Phase 3+) |
| **Version Control & CI** | Git, GitHub, GitHub Actions |
| **Target Deployment** | Docker, Azure Container Apps / AWS ECS |

---

## Directory Structure

```
Argus/
├── .github/
│   ├── workflows/
│   │   └── ci.yml               # Automated CI for frontend & backend
│   └── pull_request_template.md # PR guidelines and verification checklist
├── backend/
│   ├── app/
│   │   ├── api/                 # API routers and endpoint handlers
│   │   │   ├── health.py        # Health check route
│   │   │   └── router.py        # Centralized router mount
│   │   ├── core/                # Centralized settings & configuration
│   │   │   └── config.py        # Pydantic BaseSettings
│   │   ├── schemas/             # Pydantic validation & response models
│   │   │   └── health.py        # Health response schema
│   │   ├── services/            # Business & orchestration logic
│   │   │   └── health_service.py# Health check service implementation
│   │   └── main.py              # FastAPI app factory, CORS & error handlers
│   ├── tests/                   # Backend unit and integration tests
│   │   ├── conftest.py          # Pytest async client fixtures
│   │   └── test_health.py       # Health endpoint test suite
│   ├── .env.example             # Backend environment template
│   ├── pyproject.toml           # Project packaging & dependencies
│   ├── requirements.txt         # Core dependencies
│   └── requirements-dev.txt     # Development and testing dependencies
├── frontend/
│   ├── public/                  # Static assets & brand icons
│   ├── src/
│   │   ├── components/          # Reusable UI components (dark navy/cyan tokens)
│   │   ├── hooks/               # React custom hooks (useHealth, etc.)
│   │   ├── pages/               # Page views (HomePage foundation)
│   │   ├── services/            # API client service layer
│   │   ├── styles/              # Design tokens and Tailwind CSS v4 entrypoints
│   │   ├── types/               # TypeScript interfaces & contracts
│   │   ├── App.tsx              # Root application layout
│   │   └── main.tsx             # Application DOM mounting
│   ├── tests/                   # Frontend unit & smoke tests
│   │   ├── setup.ts             # Vitest test setup
│   │   └── App.test.tsx         # App rendering smoke test
│   ├── .env.example             # Frontend environment template
│   ├── index.html               # Main HTML entrypoint
│   ├── package.json             # NPM scripts and dependencies
│   ├── tsconfig.json            # TypeScript compiler configuration
│   └── vite.config.ts           # Vite + Tailwind v4 bundler config
├── docs/
│   ├── architecture.md          # Architectural blueprints and boundaries
│   └── development-guide.md     # Setup, testing, and contribution protocols
├── scripts/
│   ├── dev-backend.ps1 / .sh    # Convenience script: Start backend server
│   └── dev-frontend.ps1 / .sh   # Convenience script: Start frontend server
├── .gitignore                   # Comprehensive git exclusions
├── .env.example                 # Root environment template
├── LICENSE                      # MIT Open Source License
└── README.md                    # Project documentation
```

---

## Prerequisites

- **Node.js**: `v20.x` or `v22.x LTS`
- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Git**: `2.40+`

---

## Installation & Local Run Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/argus.git
cd argus
```

### 2. Backend Setup & Startup
```powershell
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Linux/macOS: source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Create environment file
cp .env.example .env

# Run FastAPI dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now running at **`http://localhost:8000`**.  
Interactive API docs: **`http://localhost:8000/api/docs`**  
Health endpoint: **`http://localhost:8000/api/health`**

### 3. Frontend Setup & Startup
```powershell
# In a separate terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Start Vite dev server
npm run dev
```
The frontend is now accessible at **`http://localhost:5173`**.

---

## Testing & Verification

### Run Backend Tests (Pytest)
```bash
cd backend
pytest -v
```

### Run Frontend Tests (Vitest)
```bash
cd frontend
npm run test
```

### Run Frontend Production Build
```bash
cd frontend
npm run build
```

---

## Planned Implementation Phases

| Phase | Title | Scope & Deliverables | Status |
| :---: | :--- | :--- | :---: |
| **Phase 1** | **Repository Foundation & Project Organization** | Clean directory structure, FastAPI foundation, health endpoint, React 19 + Tailwind v4 setup, design tokens, CI workflow, and documentation. | ✅ **Complete** |
| **Phase 2** | **Core Prompt Processing Pipeline** | Intent extraction parser, semantic constraint preservation rules, and prompt diff engine. | ⏳ Planned |
| **Phase 3** | **Local Inference Integration** | Ollama integration with `Qwen2.5-1.5B-Instruct`, compression prompting, and latency benchmarks. | ⏳ Planned |
| **Phase 4** | **Main LLM Dispatch & Provider Adapters** | Provider client integrations (OpenAI, Anthropic, Gemini), stream forwarding, and fallback handlers. | ⏳ Planned |
| **Phase 5** | **Dashboard & Optimization Telemetry** | Interactive prompt optimization playground, token savings metrics, cost estimation analytics. | ⏳ Planned |
| **Phase 6** | **Cloud Deployment & Production Hardening** | Dockerization, Azure Container Apps / AWS ECS deployment, and observability. | ⏳ Planned |

---

## Contributing & Development Guide

For detailed guidelines on branch naming, code style, and test coverage, see the [`docs/development-guide.md`](docs/development-guide.md).

---

## License

This project is licensed under the [MIT License](LICENSE).
