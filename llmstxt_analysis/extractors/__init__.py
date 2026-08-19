"""Feature extractors. Importing this package populates the registry.

To add an analysis: create a module here with a ``@register``-decorated
``RowExtractor``/``BatchExtractor`` subclass and import it below.
"""
from . import abuse, agentaccess, conformance, core, generator, language, policy, tokens  # noqa: F401

__all__ = [
    "core",
    "conformance",
    "generator",
    "agentaccess",
    "policy",
    "abuse",
    "language",
    "tokens",
]
