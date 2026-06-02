try:
    from wakewordlab._loader.loader import WkwSession
    HAS_COMPILED_LOADER = True
except ImportError:
    from wakewordlab._loader._fallback import WkwSession
    HAS_COMPILED_LOADER = False

__all__ = ["WkwSession", "HAS_COMPILED_LOADER"]
