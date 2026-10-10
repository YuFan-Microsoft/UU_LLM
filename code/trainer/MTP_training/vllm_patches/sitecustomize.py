"""Runtime patch for the vLLM hidden-state server used by MTP training (loaded through PYTHONPATH).

vLLM 0.28's ExampleHiddenStatesConnector.request_finished does `self._request_filenames.pop(req_id)`, but a request
aborted before it was ever scheduled never got a filename. Speculators aborts queued requests when an epoch ends, so
the engine died with `KeyError: 'cmpl-...'`. The patched method returns (False, None) for such requests (nothing to
save) and otherwise runs the original code unchanged.

The scheduler runs in vLLM's EngineCore subprocess(es), so the patch is installed with an import hook here, in
sitecustomize, which every Python process started with this directory on PYTHONPATH runs; the connector module is
patched when it is first imported. run_speculators_mtp.sh puts this directory on PYTHONPATH for the vLLM server only.
An existing sitecustomize further down sys.path is still loaded.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector"
_HERE = os.path.dirname(os.path.abspath(__file__))


def _patch(module) -> None:
    connector = getattr(module, "ExampleHiddenStatesConnector", None)
    original = getattr(connector, "request_finished", None)
    if original is None or getattr(original, "_mtp_training_patch", False):
        return

    def request_finished(self, request, block_ids):
        if request.request_id not in self._request_filenames:  # aborted before it was ever scheduled
            return False, None
        return original(self, request, block_ids)

    request_finished._mtp_training_patch = True
    request_finished.__doc__ = original.__doc__
    connector.request_finished = request_finished


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
else:
    sys.meta_path.insert(0, _PatchOnImport())

# Chain to the sitecustomize this file shadows, if any.
_others = [entry for entry in sys.path if os.path.abspath(entry or os.curdir) != _HERE]
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", _others)
if _spec is not None and _spec.loader is not None and _spec.origin != os.path.abspath(__file__):
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
