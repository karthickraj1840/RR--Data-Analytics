def run_pipeline() -> dict:
    from .run_all import run_pipeline as run

    return run()


__all__ = ["run_pipeline"]
