import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="roa_tests_"))
os.environ["ROA_DATA_DIR"] = str(_TMP / "data")
os.environ["ROA_RUNTIME_DIR"] = str(_TMP / "runtime")
os.environ["ROA_OTEL_ENABLED"] = "false"
os.environ["ROA_LLM_API_KEY"] = "test"

import pytest  # noqa: E402

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

from roa import telemetry  # noqa: E402
from roa.config import settings  # noqa: E402
from roa.harness import GuardedStore, load_harness  # noqa: E402

MEM = InMemorySpanExporter()
telemetry.init_tracing(exporter=MEM)


@pytest.fixture(scope="session")
def bundle():
    return load_harness(settings.harness_dir, lock=False)


@pytest.fixture()
def store(tmp_path):
    return GuardedStore(tmp_path / "runtime", settings.harness_dir)
