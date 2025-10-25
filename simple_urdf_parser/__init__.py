"""Top-level package exports for simple_urdf_parser.

Expose the common public symbols so tests and users can do:

	from simple_urdf_parser import Robot

This import intentionally forwards to the slower `parser` module. If
dependencies required by the parser (e.g. numpy, spatialmath, trimesh)
are not installed, the resulting ImportError will indicate what to
install.
"""

try:
	# Import Robot into the package namespace for convenience
	from .parser import Robot  # noqa: F401
except Exception as _ex:
	# Provide a clearer error if importing the heavy parser module fails.
	raise ImportError(
		"Could not import Robot from simple_urdf_parser.parser. "
		"Make sure you run tests from the project root or install the package (e.g. `pip install -e .`) "
		"and that package dependencies are installed.") from _ex

__all__ = ["Robot"]
