# src/hd/__init__.py
"""VisiumHD extension package.

All HD-specific code lives here and imports the SHARED, UNMODIFIED core in
`src/models` and `src/data`. Nothing under those packages is changed, so the
regular-Visium pipeline runs the original GitHub code byte-for-byte.

Importing this package registers the HD encoders into the shared registries.
"""
from . import encoders  # noqa: F401  (registers HD gene/image encoders on import)
