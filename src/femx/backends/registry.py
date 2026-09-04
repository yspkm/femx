"""Explicit backend registry with no import-time plugin discovery."""

from collections.abc import Callable

from femx.backends.protocol import Backend
from femx.core.errors import BackendUnavailableError, ContractError

BackendFactory = Callable[[], Backend]


class BackendRegistry:
    """A deterministic registry populated explicitly by applications."""

    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        """Register a unique backend factory."""

        if not name or name.strip() != name:
            raise ContractError("backend registry name must be non-empty and trimmed")
        if name in self._factories:
            raise ContractError(f"backend {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str) -> Backend:
        """Create one backend or fail without fallback."""

        try:
            factory = self._factories[name]
        except KeyError as error:
            available = ", ".join(self.names()) or "<none>"
            raise BackendUnavailableError(
                f"backend {name!r} is not registered; available: {available}"
            ) from error
        backend = factory()
        if backend.descriptor.name != name:
            raise ContractError(
                f"backend factory registered as {name!r} returned {backend.descriptor.name!r}"
            )
        return backend

    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(sorted(self._factories))
