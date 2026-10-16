# Argus Developer Guide

This guide details the local setup, development conventions, testing protocols, and verification steps for developing Argus.

---

## 1. Prerequisites

Ensure the following tools are installed on your machine:
- **Node.js**: v20+ or v22 LTS (includes `npm`)
- **Python**: v3.10 to v3.13
- **Git**: v2.40+
- **Ollama**: (Required in Phase 3 for local model execution)

---

## 2. Quickstart: Local Environment Setup

### 2.1 Backend Setup

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Create and activate a Python virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS (Bash)**:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

5. **Start the development server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Or use the convenience script from the root:
   ```powershell
   .\scripts\dev-backend.ps1
   ```

6. **Verify the backend**:
   Open [http://localhost:8000/api/health](http://localhost:8000/api/health) in your browser or curl:
   ```bash
   curl http://localhost:8000/api/health
   ```
   Interactive OpenAPI docs are available at [http://localhost:8000/api/docs](http://localhost:8000/api/docs).

---

### 2.2 Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

4. **Start Vite development server**:
   ```bash
   npm run dev
   ```
   Or use the convenience script from the root:
   ```powershell
   .\scripts\dev-frontend.ps1
   ```

5. **Access the application**:
   Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 3. Running Automated Tests

### 3.1 Backend Tests
Run the pytest test suite inside the `backend/` directory:
```bash
cd backend
pytest -v
```

### 3.2 Frontend Tests
Run Vitest unit and smoke tests in the `frontend/` directory:
```bash
cd frontend
npm run test
```

### 3.3 Frontend Production Build Validation
Verify type-checking and bundling:
```bash
cd frontend
npm run build
```

---

## 4. Coding Standards & Architectural Guardrails

### 4.1 Separation of Concerns
- **Frontend**: Presentational and interactive components only. UI components must consume backend services via `src/services/api.ts` or custom hooks.
- **Backend API Layer (`app/api/`)**: Translates HTTP requests to schema objects and calls service methods. Does not perform raw business logic.
- **Backend Schema Layer (`app/schemas/`)**: Declares Pydantic models for validation and serialization.
- **Backend Service Layer (`app/services/`)**: Orchestrates business logic, model calls, and data processing.
- **Backend Core Layer (`app/core/`)**: Houses configuration, secrets management, and cross-cutting utilities.

### 4.2 Styling & Tokens
- Use the predefined semantic tokens in `frontend/src/styles/tokens.css` and Tailwind v4 `@theme`.
- Maintain dark navy surfaces (`#060a12`, `#0b1120`, `#131e34`) and restrained cyan accents (`#06b6d4`, `#22d3ee`).
- Avoid arbitrary colors or bright flashy animations.
- Ensure all interactive controls have accessible `:focus-visible` outlines.

### 4.3 Git & Commit Hygiene
- Never commit `.env`, model weights (`.gguf`, `.safetensors`), virtual environments (`.venv`), or build artifacts (`dist/`).
- Verify CI passes before merging pull requests.
