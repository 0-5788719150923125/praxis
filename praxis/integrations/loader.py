"""Integration loader for Praxis integrations."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class IntegrationLoader:
    """Loads and manages Praxis integrations from the staging directory."""

    def __init__(self, integrations_dir: str = "./staging/integrations"):
        self.integrations_dir = Path(integrations_dir)
        self.loaded_integrations = {}
        self.installed_dependencies: Set[str] = set()  # Track installed packages
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

    def discover_integrations(self) -> List[Dict]:
        """Find all integrations in staging directory."""
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
                and (integration_dir / "integration.yaml").exists()
            ):
                try:
                    manifest_path = integration_dir / "integration.yaml"
                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)
                    manifest["path"] = integration_dir
                    integrations.append(manifest)
                    print(
                        f"[Integrations] Found integration: {manifest['name']} v{manifest.get('version', 'unknown')}"
                    )
                except Exception as e:
                    print(
                        f"[Integrations] Warning: Failed to load integration manifest {integration_dir}: {e}"
                    )

        if integrations:
            print(f"[Integrations] Discovered {len(integrations)} integration(s)")
        else:
            print(f"[Integrations] No integrations found")

        return integrations

    def load_integration(
        self, manifest: Dict, args: Optional[Any] = None, verbose: bool = True
    ) -> bool:
        """Load a single integration if conditions are met."""
        integration_name = manifest["name"]
        integration_path = manifest["path"]

        # Check conditions if args provided
        if args and not self._check_conditions(manifest.get("conditions", []), args):
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
        if not self._check_and_install_dependencies(manifest):
            if verbose:
                print(
                    f"[Integrations] Skipping {integration_name} (missing dependencies)"
                )
            return False

        try:
            # Load integration
            init_file = integration_path / "__init__.py"
            if not init_file.exists():
                print(
                    f"[Integrations] Warning: Integration {integration_name} missing __init__.py"
                )
                return False

            spec = importlib.util.spec_from_file_location(
                f"staging_{integration_name}", init_file
            )
            integration = importlib.util.module_from_spec(spec)

            # Add to sys.modules to make imports work
            sys.modules[f"staging_{integration_name}"] = integration
            spec.loader.exec_module(integration)

            self.loaded_integrations[integration_name] = integration
            registered_features = self._register_integrations(manifest, integration)

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

    def _check_and_install_dependencies(self, manifest: Dict) -> bool:
        """Check and automatically install integration dependencies."""
        dependencies = manifest.get("dependencies", {}).get("python", [])
        if not dependencies:
            return True

        integration_name = manifest.get("name", "unknown")
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

    def _register_integrations(self, manifest: Dict, integration: Any) -> List[str]:
        """Register integration's features."""
        integrations = manifest.get("integrations", {})
        registered = []
        integration_name = manifest["name"]

        # Auto-detect common functions if not explicitly configured
        provides = manifest.get("provides", [])

        # CLI integration (explicit or auto-detect)
        if "cli" in integrations:
            cli_config = integrations["cli"]
            if hasattr(integration, cli_config["function"]):
                cli_func = getattr(integration, cli_config["function"])
                self.integration_registry["cli"].append(cli_func)
                registered.append("CLI arguments")
        elif "cli_args" in provides and hasattr(integration, "add_cli_args"):
            # Auto-detect CLI function
            cli_func = getattr(integration, "add_cli_args")
            self.integration_registry["cli"].append(cli_func)
            registered.append("CLI arguments")

        # Logger integration
        if "loggers" in integrations:
            logger_config = integrations["loggers"]
            if hasattr(integration, logger_config["class"]):
                logger_class = getattr(integration, logger_config["class"])
                self.integration_registry["loggers"][manifest["name"]] = logger_class
                registered.append("logger class")

        # Dataset integration (explicit or auto-detect)
        if "datasets" in integrations:
            dataset_config = integrations["datasets"]
            if hasattr(integration, dataset_config["class"]):
                dataset_class = getattr(integration, dataset_config["class"])
                self.integration_registry["datasets"][manifest["name"]] = dataset_class
                registered.append("dataset class")
        elif "datasets" in provides and hasattr(integration, "provide_dataset"):
            # Auto-detect dataset provider function
            dataset_func = getattr(integration, "provide_dataset")
            self.integration_registry["datasets"][integration_name] = dataset_func
            registered.append("dataset provider")

        # Lifecycle integration (explicit or auto-detect)
        if "lifecycle" in integrations:
            lifecycle = integrations["lifecycle"]
            lifecycle_hooks = []
            if "init" in lifecycle and hasattr(integration, lifecycle["init"]):
                init_func = getattr(integration, lifecycle["init"])
                self.integration_registry["lifecycle"]["init"].append(init_func)
                lifecycle_hooks.append("init")
            if "cleanup" in lifecycle and hasattr(integration, lifecycle["cleanup"]):
                cleanup_func = getattr(integration, lifecycle["cleanup"])
                self.integration_registry["lifecycle"]["cleanup"].append(cleanup_func)
                lifecycle_hooks.append("cleanup")
            if lifecycle_hooks:
                registered.append(f"lifecycle ({', '.join(lifecycle_hooks)})")
        elif "lifecycle" in provides:
            # Auto-detect lifecycle functions
            lifecycle_hooks = []
            if hasattr(integration, "initialize"):
                init_func = getattr(integration, "initialize")
                self.integration_registry["lifecycle"]["init"].append(init_func)
                lifecycle_hooks.append("init")
            if hasattr(integration, "cleanup"):
                cleanup_func = getattr(integration, "cleanup")
                self.integration_registry["lifecycle"]["cleanup"].append(cleanup_func)
                lifecycle_hooks.append("cleanup")
            if lifecycle_hooks:
                registered.append(f"lifecycle ({', '.join(lifecycle_hooks)})")

        # Cleanup directories integration
        if "cleanup_dirs" in integrations:
            cleanup_dirs = integrations["cleanup_dirs"]
            if isinstance(cleanup_dirs, list):
                self.integration_registry["cleanup_dirs"].extend(cleanup_dirs)
                registered.append(f"cleanup dirs ({', '.join(cleanup_dirs)})")
            else:
                self.integration_registry["cleanup_dirs"].append(cleanup_dirs)
                registered.append(f"cleanup dir ({cleanup_dirs})")

        # Logger provider integration
        if "logger_providers" in integrations:
            provider_config = integrations["logger_providers"]
            if hasattr(integration, provider_config["function"]):
                provider_func = getattr(integration, provider_config["function"])
                self.integration_registry["logger_providers"].append(provider_func)
                registered.append("logger provider")

        # API server hooks integration
        if "api_server_hooks" in integrations:
            hook_config = integrations["api_server_hooks"]
            if hasattr(integration, hook_config["function"]):
                hook_func = getattr(integration, hook_config["function"])
                self.integration_registry["api_server_hooks"].append(hook_func)
                registered.append("API server hook")

        # Request middleware integration
        if "request_middleware" in integrations:
            middleware_config = integrations["request_middleware"]
            if hasattr(integration, middleware_config["function"]):
                middleware_func = getattr(integration, middleware_config["function"])
                self.integration_registry["request_middleware"].append(middleware_func)
                registered.append("request middleware")

        return registered

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
        for init_func in self.integration_registry["lifecycle"]["init"]:
            try:
                result = init_func(*args, **kwargs)
                if result:
                    results.update(result)
            except Exception as e:
                print(f"Integration init hook failed: {e}")
        return results

    def run_cleanup_hooks(self):
        """Run all integration cleanup hooks."""
        for cleanup_func in self.integration_registry["lifecycle"]["cleanup"]:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Integration cleanup hook failed: {e}")

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
        for provider_func in self.get_logger_providers():
            try:
                logger = provider_func(cache_dir, ckpt_path, truncated_hash, **kwargs)
                if logger:
                    return logger
            except Exception as e:
                print(f"Logger provider failed: {e}")
        return None
