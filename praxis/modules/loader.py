"""Module loader for Praxis integrations."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class ModuleLoader:
    """Loads and manages Praxis modules from the staging directory."""
    
    def __init__(self, staging_dir: str = "./staging"):
        self.staging_dir = Path(staging_dir)
        self.loaded_modules = {}
        self.installed_dependencies: Set[str] = set()  # Track installed packages
        self.integration_registry = {
            'cli': [],
            'loggers': {},
            'datasets': {},
            'lifecycle': {'init': [], 'cleanup': []},
            'cleanup_dirs': [],  # Directories to clean on reset
            'logger_providers': [],  # Functions that provide loggers
            'api_server_hooks': [],  # Functions called when API server starts
            'request_middleware': []  # Functions that modify request/response headers
        }
    
    def discover_modules(self) -> List[Dict]:
        """Find all modules in staging directory."""
        modules = []
        if not self.staging_dir.exists():
            print(f"[Modules] No staging directory found at {self.staging_dir}")
            return modules
            
        print(f"[Modules] Discovering modules in {self.staging_dir}")
        
        for module_dir in self.staging_dir.iterdir():
            if module_dir.is_dir() and (module_dir / "module.yaml").exists():
                try:
                    manifest_path = module_dir / "module.yaml"
                    with open(manifest_path) as f:
                        manifest = yaml.safe_load(f)
                    manifest['path'] = module_dir
                    modules.append(manifest)
                    print(f"[Modules] Found module: {manifest['name']} v{manifest.get('version', 'unknown')}")
                except Exception as e:
                    print(f"[Modules] Warning: Failed to load module manifest {module_dir}: {e}")
        
        if modules:
            print(f"[Modules] Discovered {len(modules)} module(s)")
        else:
            print(f"[Modules] No modules found")
            
        return modules
    
    def load_module(self, manifest: Dict, args: Optional[Any] = None, verbose: bool = True) -> bool:
        """Load a single module if conditions are met."""
        module_name = manifest['name']
        module_path = manifest['path']
        
        # Check conditions if args provided
        if args and not self._check_conditions(manifest.get('conditions', []), args):
            if verbose:
                print(f"[Modules] Skipping {module_name} (conditions not met)")
            return False
            
        # Skip if already loaded
        if module_name in self.loaded_modules:
            # If already loaded and now conditions are met, just return success
            if args and verbose:
                print(f"[Modules] ✓ {module_name} activated (conditions now met)")
            return True
        
        # Check and install dependencies if needed
        if not self._check_and_install_dependencies(manifest):
            if verbose:
                print(f"[Modules] Skipping {module_name} (missing dependencies)")
            return False
        
        try:
            # Load module
            init_file = module_path / "__init__.py"
            if not init_file.exists():
                print(f"[Modules] Warning: Module {module_name} missing __init__.py")
                return False
                
            spec = importlib.util.spec_from_file_location(
                f"staging_{module_name}", 
                init_file
            )
            module = importlib.util.module_from_spec(spec)
            
            # Add to sys.modules to make imports work
            sys.modules[f"staging_{module_name}"] = module
            spec.loader.exec_module(module)
            
            self.loaded_modules[module_name] = module
            integrations = self._register_integrations(manifest, module)
            
            if verbose:
                print(f"[Modules] ✓ Loaded {module_name} from {module_path}")
                if integrations:
                    print(f"[Modules]   Provides: {', '.join(integrations)}")
            return True
            
        except Exception as e:
            print(f"[Modules] ✗ Failed to load {module_name}: {e}")
            return False
    
    def _check_conditions(self, conditions: List[str], args: Any) -> bool:
        """Check if module conditions are met."""
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
        """Check and automatically install module dependencies."""
        dependencies = manifest.get('dependencies', {}).get('python', [])
        if not dependencies:
            return True
        
        module_name = manifest.get('name', 'unknown')
        missing_deps = []
        
        # Check which dependencies are missing
        for dep in dependencies:
            # Extract package name (handle cases like 'package>=1.0.0')
            pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
            if pkg_name in self.installed_dependencies:
                continue
                
            try:
                __import__(pkg_name.replace('-', '_'))
                self.installed_dependencies.add(pkg_name)
            except ImportError:
                missing_deps.append(dep)
        
        if not missing_deps:
            return True
        
        print(f"[Modules] Module '{module_name}' requires: {', '.join(missing_deps)}")
        
        # Auto-install missing dependencies
        try:
            print(f"[Modules] Installing: {', '.join(missing_deps)}")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--quiet'
            ] + missing_deps)
            
            # Mark as installed
            for dep in missing_deps:
                pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].strip()
                self.installed_dependencies.add(pkg_name)
            
            print(f"[Modules] ✓ Successfully installed dependencies for {module_name}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[Modules] ✗ Failed to install dependencies: {e}")
            return False
    
    def _register_integrations(self, manifest: Dict, module: Any) -> List[str]:
        """Register module's integration points."""
        integrations = manifest.get('integrations', {})
        registered = []
        
        # CLI integration
        if 'cli' in integrations:
            cli_config = integrations['cli']
            if hasattr(module, cli_config['function']):
                cli_func = getattr(module, cli_config['function'])
                self.integration_registry['cli'].append(cli_func)
                registered.append('CLI arguments')
        
        # Logger integration  
        if 'loggers' in integrations:
            logger_config = integrations['loggers']
            if hasattr(module, logger_config['class']):
                logger_class = getattr(module, logger_config['class'])
                self.integration_registry['loggers'][manifest['name']] = logger_class
                registered.append('logger class')
                
        # Dataset integration
        if 'datasets' in integrations:
            dataset_config = integrations['datasets']
            if hasattr(module, dataset_config['class']):
                dataset_class = getattr(module, dataset_config['class'])
                self.integration_registry['datasets'][manifest['name']] = dataset_class
                registered.append('dataset class')
            
        # Lifecycle integration
        if 'lifecycle' in integrations:
            lifecycle = integrations['lifecycle']
            lifecycle_hooks = []
            if 'init' in lifecycle and hasattr(module, lifecycle['init']):
                init_func = getattr(module, lifecycle['init'])
                self.integration_registry['lifecycle']['init'].append(init_func)
                lifecycle_hooks.append('init')
            if 'cleanup' in lifecycle and hasattr(module, lifecycle['cleanup']):
                cleanup_func = getattr(module, lifecycle['cleanup'])
                self.integration_registry['lifecycle']['cleanup'].append(cleanup_func)
                lifecycle_hooks.append('cleanup')
            if lifecycle_hooks:
                registered.append(f"lifecycle ({', '.join(lifecycle_hooks)})")
        
        # Cleanup directories integration
        if 'cleanup_dirs' in integrations:
            cleanup_dirs = integrations['cleanup_dirs']
            if isinstance(cleanup_dirs, list):
                self.integration_registry['cleanup_dirs'].extend(cleanup_dirs)
                registered.append(f"cleanup dirs ({', '.join(cleanup_dirs)})")
            else:
                self.integration_registry['cleanup_dirs'].append(cleanup_dirs)
                registered.append(f"cleanup dir ({cleanup_dirs})")
        
        # Logger provider integration
        if 'logger_providers' in integrations:
            provider_config = integrations['logger_providers']
            if hasattr(module, provider_config['function']):
                provider_func = getattr(module, provider_config['function'])
                self.integration_registry['logger_providers'].append(provider_func)
                registered.append('logger provider')
        
        # API server hooks integration
        if 'api_server_hooks' in integrations:
            hook_config = integrations['api_server_hooks']
            if hasattr(module, hook_config['function']):
                hook_func = getattr(module, hook_config['function'])
                self.integration_registry['api_server_hooks'].append(hook_func)
                registered.append('API server hook')
        
        # Request middleware integration
        if 'request_middleware' in integrations:
            middleware_config = integrations['request_middleware']
            if hasattr(module, middleware_config['function']):
                middleware_func = getattr(module, middleware_config['function'])
                self.integration_registry['request_middleware'].append(middleware_func)
                registered.append('request middleware')
        
        return registered
    
    def get_cli_functions(self) -> List:
        """Get all CLI argument functions."""
        return self.integration_registry['cli']
    
    def get_logger(self, name: str):
        """Get logger class by name."""
        return self.integration_registry['loggers'].get(name)
    
    def get_dataset(self, name: str):
        """Get dataset class by name."""
        return self.integration_registry['datasets'].get(name)
    
    def run_init_hooks(self, *args, **kwargs):
        """Run all module initialization hooks."""
        results = {}
        for init_func in self.integration_registry['lifecycle']['init']:
            try:
                result = init_func(*args, **kwargs)
                if result:
                    results.update(result)
            except Exception as e:
                print(f"Module init hook failed: {e}")
        return results
    
    def run_cleanup_hooks(self):
        """Run all module cleanup hooks."""
        for cleanup_func in self.integration_registry['lifecycle']['cleanup']:
            try:
                cleanup_func()
            except Exception as e:
                print(f"Module cleanup hook failed: {e}")
    
    def get_cleanup_directories(self) -> List[str]:
        """Get all directories that should be cleaned on reset."""
        return self.integration_registry['cleanup_dirs']
    
    def get_logger_providers(self) -> List:
        """Get all logger provider functions."""
        return self.integration_registry['logger_providers']
    
    def get_api_server_hooks(self) -> List:
        """Get all API server hook functions."""
        return self.integration_registry['api_server_hooks']
    
    def get_request_middleware(self) -> List:
        """Get all request middleware functions."""
        return self.integration_registry['request_middleware']
    
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