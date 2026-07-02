# tests/conftest.py
#
# embedder.py does heavy, network-dependent work at import time (loading a
# SentenceTransformer model). The extraction logic we actually want to test
# (extract_functions_from_python / _js_ts / _java) is pure and has nothing to
# do with embeddings, so we mock out the ML dependencies here purely to make
# `import embedder` possible in a fast, offline, CI-friendly test run.
import sys
import types


def _install_fake_module(name, attrs=None):
    if name in sys.modules:
        return
    fake = types.ModuleType(name)
    for attr_name, attr_value in (attrs or {}).items():
        setattr(fake, attr_name, attr_value)
    sys.modules[name] = fake


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, *args, **kwargs):
        return []


_st_module = types.ModuleType("sentence_transformers")
_st_module.SentenceTransformer = _FakeSentenceTransformer
sys.modules.setdefault("sentence_transformers", _st_module)

_install_fake_module("faiss")
_tqdm_module = types.ModuleType("tqdm")
_tqdm_module.tqdm = lambda x, **kwargs: x
sys.modules.setdefault("tqdm", _tqdm_module)