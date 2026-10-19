"""Independent runtime; importing the catalog does not import accelerator libraries."""

from .models import catalog

__all__ = ["catalog"]
