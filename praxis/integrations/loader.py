"""Integration loader for Praxis integrations."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BaseIntegration, IntegrationFactory, IntegrationSpec


class IntegrationLoader:
    """Loads and manages Praxis integrations."""

    def __init__(self, integrations_dir: str = "./integrations"):
        self.integrations_dir = Path(integrations_dir)
        self.loaded_integrations: Dict[str, BaseIntegration] = {}
        self.installed_dependencies: Set[str] = set()  # Track installed packages
        self.available_specs: Dict[str, IntegrationSpec] = (
            {}
        )  # Track all discovered integrations
        self.used_integrations: Set[str] = set()  # Track which are actually being used

        # Legacy registry for backward compatibility
        self.integration_registry = {
            "cli": [],
            "loggers": {},
            "datasets": {},
            "lifecycle": {"init": [], "cleanup": []},
            "cleanup_dirs": [],  # Directories to clean on reset
            "logger_providers": [],  # Functions that provide loggers
            "api_server_hooks": [],  # Functions called when API server starts
            "request_middleware": [],  # Functions that modify request/response headers
        }

    def discover_integrations(self) -> List[IntegrationSpec]:
        """Find all integrations in integrations directory.

        Returns:
            List of IntegrationSpec objects
        """
        integrations = []
        if not self.integrations_dir.exists():
            # Silent - directory doesn't exist
            return integrations

        for integration_dir in self.integrations_dir.iterdir():
            if integration_dir.is_dir() and (integration_dir / "spec.yaml").exists():
                try:
                    spec_path = integration_dir / "spec.yaml"
                    spec = IntegrationSpec.from_file(spec_path)
                    integrations.append(spec)
                    self.available_specs[spec.name] = spec
                except Exception:
                    # Silent - failed to load spec
                    pass

        return integrations

    def load_integration(
        self, spec: IntegrationSpec, args: Optional[Any] = None, verbose: bool = True
    ) -> bool:
        """Load a single integration if conditions are met.

        Args:
            spec: Integration specification
            args: Command-line arguments
            verbose: Whether to print verbose output

        Returns:
            True if integration was loaded successfully
        """
        integration_name = spec.name
        integration_path = spec.path

        # Check conditions if args provided
        if args and not self._check_conditions(spec.conditions, args):
            return False

        # Skip if already loaded
        if integration_name in self.loaded_integrations:
            # Mark as used if conditions are met
            if args:
                self.used_integrations.add(integration_name)
            return True

        # Check and install dependencies if needed (silent)
        if not self._check_and_install_dependencies(spec, verbose=False):
            return False

        try:
            # Load integration module
            init_file = integration_path / "__init__.py"
            if not init_file.exists():
                return False

            spec_name = importlib.util.spec_from_file_location(
                f"staging_{integration_name}", init_file
            )
            module = importlib.util.module_from_spec(spec_name)

            # Add to sys.modules to make imports work
            sys.modules[f"staging_{integration_name}"] = module
            spec_name.loader.exec_module(module)

            # Create integration instance using factory
            integration = IntegrationFactory.create_from_module(module, spec)
            self.loaded_integrations[integration_name] = integration

            # Register integration features (for backward compatibility)
            registered_features = self._register_integration(integration)

            # Mark as used if conditions are met
            if args:
                self.used_integrations.add(integration_name)

            return True

        except Exception:
            # Silent - failed to load
            return False

    def _check_conditions(self, conditions: List[str], args: Any) -> bool:
        """Check if integration conditions are met."""
        if not conditions:
            return True

        for condition in conditions:
            try:
                # Simple eval with args namespace
                if not eval(condition, {"args": args}):
                    return False
            except Exception:
                return False
        return True

    def _check_and_install_dependencies(
        self, spec: IntegrationSpec, verbose: bool = True
    ) -> bool:
        """Check and automatically install integration dependencies."""
        # First check for pyproject.toml
        pyproject_path = spec.path / "pyproject.toml"
        if pyproject_path.exists():
            return self._install_from_pyproject(spec, pyproject_path, verbose)
        
        # Fall back to spec.yaml dependencies
        dependencies = spec.dependencies.get("python", [])
        if not dependencies:
            return True

        integration_name = spec.name
        missing_deps = []

        # Check which dependencies are missing
        for dep in dependencies:
            # Extract package name (handle cases like 'package>=1.0.0')
            pkg_name = (
                dep.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
            )
            if pkg_name in self.installed_dependencies:
                continue

            try:
                __import__(pkg_name.replace("-", "_"))
                self.installed_dependencies.add(pkg_name)
            except ImportError:
                missing_deps.append(dep)

        if not missing_deps:
            return True

        # Auto-install missing dependencies (silently)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing_deps,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Mark as installed
            for dep in missing_deps:
                pkg_name = (
                    dep.split(">=")[0]
                    .split("==")[0]
                    .split("<")[0]
                    .split(">")[0]
                    .strip()
                )
                self.installed_dependencies.add(pkg_name)

            return True

        except subprocess.CalledProcessError:
            return False
    
    def _install_from_pyproject(
        self, spec: IntegrationSpec, pyproject_path: Path, verbose: bool = True
    ) -> bool:
        """Install integration using pyproject.toml."""
        try:
            # Check if integration directory is already installed as editable
            integration_name = spec.name
            check_result = subprocess.run(
                [sys.executable, "-m", "pip", "show", f"praxis-integration-{integration_name}"],
                capture_output=True,
                text=True
            )
            
            if check_result.returncode == 0:
                # Already installed
                return True
            
            # Install the integration directory as editable package
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-e", str(spec.path), "--quiet"],
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.DEVNULL if not verbose else None,
            )
            
            return True
            
        except subprocess.CalledProcessError:
            return False

    def _register_integration(self, integration: BaseIntegration) -> List[str]:
        """Register integration's features in the legacy registry.

        This maintains backward compatibility with the old registry system.
        """
        registered = []
        spec = integration.spec
        integration_name = spec.name

        # Register CLI functions
        if hasattr(integration, "add_cli_args"):
            # Create a wrapper to match old signature
            self.integration_registry["cli"].append(integration.add_cli_args)
            if spec.integrations.get("cli") or "cli_args" in spec.provides:
                registered.append("CLI arguments")

        # Register lifecycle hooks
        if hasattr(integration, "initialize"):
            init_wrapper = lambda *args, **kwargs: integration.initialize(
                *args, **kwargs
            )
            self.integration_registry["lifecycle"]["init"].append(init_wrapper)
            if "lifecycle" in spec.provides or spec.integrations.get("lifecycle"):
                registered.append("lifecycle (init)")

        if hasattr(integration, "cleanup"):
            self.integration_registry["lifecycle"]["cleanup"].append(
                integration.cleanup
            )
            if "lifecycle" in spec.provides or spec.integrations.get("lifecycle"):
                if "lifecycle (init)" in registered:
                    registered[-1] = "lifecycle (init, cleanup)"
                else:
                    registered.append("lifecycle (cleanup)")

        # Register dataset providers
        if hasattr(integration, "provide_dataset"):
            dataset_wrapper = lambda *args, **kwargs: integration.provide_dataset(
                *args, **kwargs
            )
            self.integration_registry["datasets"][integration_name] = dataset_wrapper
            if "datasets" in spec.provides or spec.integrations.get("datasets"):
                registered.append("dataset provider")

        # Register logger providers
        if hasattr(integration, "provide_logger"):
            logger_wrapper = lambda *args, **kwargs: integration.provide_logger(
                *args, **kwargs
            )
            self.integration_registry["logger_providers"].append(logger_wrapper)
            if "loggers" in spec.provides or spec.integrations.get("logger_providers"):
                registered.append("logger provider")

        # Register API server hooks
        if hasattr(integration, "api_server_hook"):
            self.integration_registry["api_server_hooks"].append(
                integration.api_server_hook
            )
            registered.append("API server hook")
        elif hasattr(integration, "on_api_server_start"):
            self.integration_registry["api_server_hooks"].append(
                integration.on_api_server_start
            )
            registered.append("API server hook")

        # Register request middleware
        if hasattr(integration, "request_middleware"):
            self.integration_registry["request_middleware"].append(
                integration.request_middleware
            )
            if spec.integrations.get("request_middleware"):
                registered.append("request middleware")

        # Register cleanup directories
        cleanup_dirs = spec.integrations.get("cleanup_dirs", [])
        if cleanup_dirs:
            if isinstance(cleanup_dirs, list):
                self.integration_registry["cleanup_dirs"].extend(cleanup_dirs)
                registered.append(f"cleanup dirs ({', '.join(cleanup_dirs)})")
            else:
                self.integration_registry["cleanup_dirs"].append(cleanup_dirs)
                registered.append(f"cleanup dir ({cleanup_dirs})")

        return registered

    # Legacy compatibility methods
    def get_cli_functions(self) -> List:
        """Get all CLI argument functions."""
        return self.integration_registry["cli"]

    def get_logger(self, name: str):
        """Get logger class by name."""
        return self.integration_registry["loggers"].get(name)

    def get_dataset(self, name: str):
        """Get dataset class by name."""
        return self.integration_registry["datasets"].get(name)

    def run_init_hooks(self, *args, **kwargs):
        """Run all integration initialization hooks."""
        results = {}
        for integration in self.loaded_integrations.values():
            try:
                result = integration.initialize(*args, **kwargs)
                if result:
                    results.update(result)
            except Exception as e:
                print(f"Integration init hook failed for {integration.name}: {e}")
        return results

    def run_cleanup_hooks(self):
        """Run all integration cleanup hooks."""
        for integration in self.loaded_integrations.values():
            try:
                integration.cleanup()
            except Exception as e:
                print(f"Integration cleanup hook failed for {integration.name}: {e}")

    def get_cleanup_directories(self) -> List[str]:
        """Get all directories that should be cleaned on reset."""
        return self.integration_registry["cleanup_dirs"]

    def get_logger_providers(self) -> List:
        """Get all logger provider functions."""
        return self.integration_registry["logger_providers"]

    def get_api_server_hooks(self) -> List:
        """Get all API server hook functions."""
        return self.integration_registry["api_server_hooks"]

    def get_request_middleware(self) -> List:
        """Get all request middleware functions."""
        return self.integration_registry["request_middleware"]

    def create_logger(self, cache_dir, ckpt_path=None, truncated_hash=None, **kwargs):
        """Create logger using loaded modules."""
        for integration in self.loaded_integrations.values():
            try:
                logger = integration.provide_logger(
                    cache_dir, ckpt_path, truncated_hash, **kwargs
                )
                if logger:
                    return logger
            except Exception as e:
                print(f"Logger provider failed for {integration.name}: {e}")
        return None

    def print_summary(self):
        """Print a clean summary of available and used integrations."""
        # Get available integrations (not used)
        available = [
            name
            for name in self.available_specs.keys()
            if name not in self.used_integrations
        ]

        # Get used integrations
        used = list(self.used_integrations)

        # Print summary
        if available or used:
            print(f"[Integrations] Path: {self.integrations_dir}")

            if available:
                print(f"[Integrations] Available: {', '.join(sorted(available))}")

            if used:
                print(f"[Integrations] Loaded: {', '.join(sorted(used))}")
