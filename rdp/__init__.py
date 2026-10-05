"""rdp: a small research data platform for LLM agent experiments."""

from rdp.contracts import CheckResult, ContractViolation, check_run, no_overlap
from rdp.datasets import Registry
from rdp.metrics import Run, start_run
from rdp.query import connect
from rdp.schemas import SCHEMA_VERSION
from rdp.store import SchemaError, Store

__all__ = [
    "SCHEMA_VERSION", "CheckResult", "ContractViolation", "Registry", "Run", "SchemaError",
    "Store", "check_run", "connect", "no_overlap", "start_run",
]
