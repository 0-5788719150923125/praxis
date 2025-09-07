"""Base integration class and validation for Praxis integrations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class IntegrationSpec:
    """Specification/manifest for an integration."""

    REQUIRED_FIELDS = ["name", "version", "description"]
    OPTIONAL_FIELDS = [
        "author",
        "dependencies",
        "conditions",
        "provides",
        "integrations",
    ]

    def __init__(self, spec_dict: Dict[str, Any], path: Path):
        """Initialize integration spec from dictionary and path.
        
        Args:
            spec_dict: Dictionary containing the spec data
            path: Path to the integration directory
        """
        self.path = path
        self._data = spec_dict
        self.validate()

    def validate(self):
        """Validate the integration spec has required fields."""
        for field in self.REQUIRED_FIELDS:
            if field not in self._data:
                raise ValueError(
                    f"Integration spec at {self.path} missing required field: {field}"
                )

        # Validate version format (should be semantic versioning)
        version = self._data["version"]
        if not isinstance(version, str) or not any(
            c.isdigit() for c in version
        ):
            raise ValueError(
                f"Invalid version format '{version}' in {self.path}. "
                "Expected semantic versioning (e.g., '1.0.0')"
            )

    @property
    def name(self) -> str:
        """Get integration name."""
        return self._data["name"]

    @property
    def version(self) -> str:
        """Get integration version."""
        return self._data["version"]

    @property
    def description(self) -> str:
        """Get integration description."""
        return self._data["description"]

    @property
    def author(self) -> Optional[str]:
        """Get integration author."""
        return self._data.get("author")

    @property
    def dependencies(self) -> Dict[str, List[str]]:
        """Get integration dependencies."""
        return self._data.get("dependencies", {})

    @property
    def conditions(self) -> List[str]:
        """Get integration conditions."""
        return self._data.get("conditions", [])

    @property
    def provides(self) -> List[str]:
        """Get list of features this integration provides."""
        return self._data.get("provides", [])

    @property
    def integrations(self) -> Dict[str, Any]:
        """Get integration configuration."""
        return self._data.get("integrations", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert spec to dictionary."""
        result = dict(self._data)
        result["path"] = self.path
        return result

    @classmethod
    def from_file(cls, spec_path: Path) -> "IntegrationSpec":
        """Load integration spec from YAML file.
        
        Args:
            spec_path: Path to the spec.yaml file
            
        Returns:
            IntegrationSpec instance
        """
        with open(spec_path) as f:
            spec_dict = yaml.safe_load(f)
        return cls(spec_dict, spec_path.parent)


class BaseIntegration(ABC):
    """Abstract base class for all Praxis integrations.
    
    All integrations should extend this class and implement the required methods.
    This ensures a consistent interface across all integrations.
    """

    def __init__(self, spec: IntegrationSpec):
        """Initialize the integration with its specification.
        
        Args:
            spec: The integration specification
        """
        self.spec = spec
        self._initialized = False

    @property
    def name(self) -> str:
        """Get the integration name."""
        return self.spec.name

    @property
    def version(self) -> str:
        """Get the integration version."""
        return self.spec.version

    def add_cli_args(self, parser) -> None:
        """Add CLI arguments for this integration.
        
        Args:
            parser: ArgumentParser to add arguments to
            
        Note: Override this method if your integration adds CLI arguments.
        """
        pass

    def initialize(
        self, args: Any, cache_dir: str, ckpt_path: Optional[str] = None, 
        truncated_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize the integration when conditions are met.
        
        Args:
            args: Parsed command-line arguments
            cache_dir: Cache directory path
            ckpt_path: Optional checkpoint path
            truncated_hash: Optional truncated hash
            
        Returns:
            Dictionary of initialization results (can be empty)
            
        Note: Override this method if your integration needs initialization.
        """
        self._initialized = True
        return {}

    def cleanup(self) -> None:
        """Clean up integration resources.
        
        Note: Override this method if your integration needs cleanup.
        """
        pass

    def provide_dataset(
        self, tokenizer: Any, seed: int, config: Optional[Any] = None, *args
    ) -> Optional[Any]:
        """Provide a dataset for training.
        
        Args:
            tokenizer: The tokenizer to use
            seed: Random seed
            config: Optional configuration
            *args: Additional arguments
            
        Returns:
            Dataset instance or None if not applicable
            
        Note: Override this method if your integration provides datasets.
        """
        return None

    def provide_logger(
        self, cache_dir: str, ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None, **kwargs
    ) -> Optional[Any]:
        """Provide a logger for training.
        
        Args:
            cache_dir: Cache directory path
            ckpt_path: Optional checkpoint path
            truncated_hash: Optional truncated hash
            **kwargs: Additional keyword arguments
            
        Returns:
            Logger instance or None if not applicable
            
        Note: Override this method if your integration provides loggers.
        """
        return None

    def on_api_server_start(self, app: Any, args: Any) -> None:
        """Hook called when API server starts.
        
        Args:
            app: Flask application instance
            args: Command-line arguments
            
        Note: Override this method if your integration needs API server hooks.
        """
        pass

    def request_middleware(self, request: Any, response: Any = None) -> None:
        """Middleware for modifying requests/responses.
        
        Args:
            request: The request object
            response: The response object (None for before_request phase)
            
        Note: Override this method if your integration provides middleware.
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if the integration has been initialized."""
        return self._initialized

    def __repr__(self) -> str:
        """String representation of the integration."""
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"


class IntegrationFactory:
    """Factory for creating integration instances from modules."""

    @staticmethod
    def create_from_module(module: Any, spec: IntegrationSpec) -> BaseIntegration:
        """Create an integration instance from a module.
        
        Args:
            module: The loaded Python module
            spec: The integration specification
            
        Returns:
            Integration instance
            
        Raises:
            ValueError: If module doesn't have an Integration class
        """
        # Look for an Integration class in the module
        if hasattr(module, "Integration"):
            integration_class = getattr(module, "Integration")
            if not issubclass(integration_class, BaseIntegration):
                raise ValueError(
                    f"Integration class in {spec.name} must extend BaseIntegration"
                )
            return integration_class(spec)
        
        # Fallback: create a dynamic integration wrapper for legacy modules
        return LegacyIntegrationWrapper(module, spec)


class LegacyIntegrationWrapper(BaseIntegration):
    """Wrapper for legacy integrations that don't extend BaseIntegration."""

    def __init__(self, module: Any, spec: IntegrationSpec):
        """Initialize wrapper with legacy module.
        
        Args:
            module: The legacy integration module
            spec: The integration specification
        """
        super().__init__(spec)
        self.module = module

    def add_cli_args(self, parser) -> None:
        """Delegate to legacy module's add_cli_args if it exists."""
        if hasattr(self.module, "add_cli_args"):
            self.module.add_cli_args(parser)

    def initialize(
        self, args: Any, cache_dir: str, ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delegate to legacy module's initialize if it exists."""
        if hasattr(self.module, "initialize"):
            result = self.module.initialize(args, cache_dir, ckpt_path, truncated_hash)
            self._initialized = True
            return result or {}
        self._initialized = True
        return {}

    def cleanup(self) -> None:
        """Delegate to legacy module's cleanup if it exists."""
        if hasattr(self.module, "cleanup"):
            self.module.cleanup()

    def provide_dataset(
        self, tokenizer: Any, seed: int, config: Optional[Any] = None, *args
    ) -> Optional[Any]:
        """Delegate to legacy module's provide_dataset if it exists."""
        if hasattr(self.module, "provide_dataset"):
            return self.module.provide_dataset(tokenizer, seed, config, *args)
        return None

    def provide_logger(
        self, cache_dir: str, ckpt_path: Optional[str] = None,
        truncated_hash: Optional[str] = None, **kwargs
    ) -> Optional[Any]:
        """Delegate to legacy module's provide_logger if it exists."""
        if hasattr(self.module, "provide_logger"):
            return self.module.provide_logger(cache_dir, ckpt_path, truncated_hash, **kwargs)
        return None

    def on_api_server_start(self, app: Any, args: Any) -> None:
        """Delegate to legacy module's on_api_server_start if it exists."""
        if hasattr(self.module, "on_api_server_start"):
            self.module.on_api_server_start(app, args)

    def request_middleware(self, request: Any, response: Any = None) -> None:
        """Delegate to legacy module's request_middleware if it exists."""
        if hasattr(self.module, "request_middleware"):
            return self.module.request_middleware(request, response)