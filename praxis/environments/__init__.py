"""
Environment Features System

This module provides a registry for environment-aware feature flags that control
runtime behavior without changing model architecture. Features are set from
environment configurations and can be queried throughout the codebase.

Example:
    from praxis.environments import EnvironmentFeatures

    if EnvironmentFeatures.is_enabled('skip_compilation'):
        # Skip torch.compile in development
        return model
"""

from typing import Any, Dict, Optional


class EnvironmentFeatures:
    """Registry for environment-aware feature flags.

    This class maintains a global registry of feature flags that are set
    from environment configurations. Features control runtime behavior
    like compilation, debugging, and data loading.
    """

    _features: Dict[str, Any] = {}
    _active_environment: Optional[str] = None

    @classmethod
    def get(cls, feature_name: str, default: Any = False) -> Any:
        """Get feature flag value.

        Args:
            feature_name: Name of the feature to retrieve
            default: Default value if feature is not set

        Returns:
            The feature value or default if not set
        """
        return cls._features.get(feature_name, default)

    @classmethod
    def set_from_environment(
        cls, features: Dict[str, Any], environment_name: Optional[str] = None
    ):
        """Set features from environment config.

        Args:
            features: Dictionary of feature names to values
            environment_name: Name of the active environment (for debugging)
        """
        cls._features.update(features)
        cls._active_environment = environment_name

    @classmethod
    def is_enabled(cls, feature: str) -> bool:
        """Check if a feature is enabled.

        Args:
            feature: Name of the feature to check

        Returns:
            True if the feature is enabled (truthy), False otherwise
        """
        return bool(cls.get(feature, False))

    @classmethod
    def clear(cls):
        """Clear all features (useful for testing)."""
        cls._features.clear()
        cls._active_environment = None

    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all currently set features.

        Returns:
            Dictionary of all feature flags
        """
        return cls._features.copy()

    @classmethod
    def get_active_environment(cls) -> Optional[str]:
        """Get the name of the active environment.

        Returns:
            Name of the active environment or None if no environment is active
        """
        return cls._active_environment

    @classmethod
    def has_feature(cls, feature: str) -> bool:
        """Check if a feature has been explicitly set.

        Args:
            feature: Name of the feature to check

        Returns:
            True if the feature has been set (regardless of value)
        """
        return feature in cls._features


# Export the main class
__all__ = ["EnvironmentFeatures"]
