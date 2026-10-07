"""`python -m dashboard` starts the observability dashboard on port 8200 (OTLP receiver at /v1/traces)."""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("dashboard.server:app", host="0.0.0.0", port=int(os.environ.get("DASHBOARD_PORT", "8200")),
                log_level="warning")
