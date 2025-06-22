try:
    from .ae_models import ConvAE, DeepConvAE
    __all__ = ["ConvAE", "DeepConvAE"]
except Exception:  # torch may be missing
    __all__ = []

