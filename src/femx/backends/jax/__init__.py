"""Native JAX FEM backend namespace.

Importing this package does not import JAX or probe devices. Concrete steady, coupled, and port
backends live in explicit submodules so callers choose when the optional runtime is imported.
"""
