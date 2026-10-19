"""Model-independent diagnostics; these reports never pass certification gates."""
from .capture import DiagnosticWriter, ModuleCapture, discover_modules
from .locate import diagnose
from .report import run_matrix
from .replay import ReplayEndpoint, replay_boundary

__all__ = ["DiagnosticWriter", "ModuleCapture", "discover_modules", "diagnose",
           "run_matrix", "ReplayEndpoint", "replay_boundary"]
