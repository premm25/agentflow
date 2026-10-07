"""`python -m roa` starts the single orchestrator process (planner + all agents + validation + reporting)."""

import uvicorn

from roa.config import settings

if __name__ == "__main__":
    uvicorn.run("roa.api:app", host=settings.host, port=settings.port, log_level="warning")
