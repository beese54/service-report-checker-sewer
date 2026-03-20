# pipeline package
#
# Re-export run_pipeline from the top-level orchestrator (pipeline.py).
# We use importlib because pipeline.py shares its name with this package,
# so a normal `import pipeline` resolves to this package, not the .py file.
import importlib.util
import os

_orchestrator = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline.py")
_spec = importlib.util.spec_from_file_location("_pipeline_orchestrator", _orchestrator)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_pipeline = _mod.run_pipeline
