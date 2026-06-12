"""ehrextract -- structured feature extraction from clinical notes."""

import logging

from ehrextract._version import __version__
from ehrextract.pipeline import Extractor, ExtractionResult, FieldError, extract
from ehrextract.providers import GenerationConfig, load_provider
from ehrextract.schema import FieldSpec, Schema, SchemaError, Task, load_schema, load_task

# Library is silent by default. CLI users get a handler installed by cli.py;
# library users configure their own root logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Extractor",
    "ExtractionResult",
    "FieldError",
    "FieldSpec",
    "GenerationConfig",
    "Schema",
    "SchemaError",
    "Task",
    "__version__",
    "extract",
    "load_provider",
    "load_schema",
    "load_task",
]
