"""Integration loader for Praxis integrations."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BaseIntegration, IntegrationFactory, IntegrationSpec


class IntegrationLoader:
    """Loads and manages Praxis integrations from the staging directory."""

    def __init__(self, integrations_dir: str = "./staging/integrations"):
        self.integrations_dir = Path(integrations_dir)
        self.loaded_integrations: Dict[str, BaseIntegration] = {}
        self.installed_dependencies: Set[str] = set()  # Track installed packages
        
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
        """Find all integrations in staging directory.
        
        Returns:
            List of IntegrationSpec objects
        """
        integrations = []
        if not self.integrations_dir.exists():
            print(
                f"[Integrations] No staging directory found at {self.integrations_dir}"
            )
            return integrations

        print(f"[Integrations] Discovering integrations in {self.integrations_dir}")

        for integration_dir in self.integrations_dir.iterdir():
            if (
                integration_dir.is_dir()
                and (integration_dir / "spec.yaml").exists()
            ):
                try:
                    spec_path = integration_dir / "spec.yaml"
                    spec = IntegrationSpec.from_file(spec_path)
                    integrations.append(spec)
                    print(
                        f"[Integrations] Found integration: {spec.name} v{spec.version}"
                    )
                except Exception as e:
                    print(
                        f"[Integrations] Warning: Failed to load integration spec {integration_dir}: {e}"
                    )

        if integrations:
            print(f"[Integrations] Discovered {len(integrations)} integration(s)")
        else:
            print(f"[Integrations] No integrations found")

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
            if verbose:
                print(
                    f"[Integrations] Skipping {integration_name} (conditions not met)"
                )
            return False

        # Skip if already loaded
        if integration_name in self.loaded_integrations:
            # If already loaded and now conditions are met, just return success
            if args and verbose:
                print(
                    f"[Integrations] ✓ {integration_name} activated (conditions now met)"
                )
            return True

        # Check and install dependencies if needed
        if not self._check_and_install_dependencies(spec):
            if verbose:
                print(
                    f"[Integrations] Skipping {integration_name} (missing dependencies)"
                )
            return False

        try:
            # Load integration module
            init_file = integration_path / "__init__.py"
            if not init_file.exists():
                print(
                    f"[Integrations] Warning: Integration {integration_name} missing __init__.py"
                )
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

            if verbose:
                print(
                    f"[Integrations] ✓ Loaded {integration_name} from {integration_path}"
                )
                if registered_features:
                    print(
                        f"[Integrations]   Provides: {', '.join(registered_features)}"
                    )
            return True

        except Exception as e:
            print(f"[Integrations] ✗ Failed to load {integration_name}: {e}")
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

    def _check_and_install_dependencies(self, spec: IntegrationSpec) -> bool:
        """Check and automatically install integration dependencies."""
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

        print(
            f"[Integrations] Integration '{integration_name}' requires: {', '.join(missing_deps)}"
        )

        # Auto-install missing dependencies
        try:
            print(f"[Integrations] Installing: {', '.join(missing_deps)}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing_deps
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

            print(
                f"[Integrations] ✓ Successfully installed dependencies for {integration_name}"
            )
            return True

        except subprocess.CalledProcessError as e:
            print(f"[Integrations] ✗ Failed to install dependencies: {e}")
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
            init_wrapper = lambda *args, **kwargs: integration.initialize(*args, **kwargs)
            self.integration_registry["lifecycle"]["init"].append(init_wrapper)
            if "lifecycle" in spec.provides or spec.integrations.get("lifecycle"):
                registered.append("lifecycle (init)")

        if hasattr(integration, "cleanup"):
            self.integration_registry["lifecycle"]["cleanup"].append(integration.cleanup)
            if "lifecycle" in spec.provides or spec.integrations.get("lifecycle"):
                if "lifecycle (init)" in registered:
                    registered[-1] = "lifecycle (init, cleanup)"
                else:
                    registered.append("lifecycle (cleanup)")

        # Register dataset providers
        if hasattr(integration, "provide_dataset"):
            dataset_wrapper = lambda *args, **kwargs: integration.provide_dataset(*args, **kwargs)
            self.integration_registry["datasets"][integration_name] = dataset_wrapper
            if "datasets" in spec.provides or spec.integrations.get("datasets"):
                registered.append("dataset provider")

        # Register logger providers
        if hasattr(integration, "provide_logger"):
            logger_wrapper = lambda *args, **kwargs: integration.provide_logger(*args, **kwargs)
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