"""Startup of the rank-0 web services: dashboard, frontend, API server."""

import importlib
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Services:
    """Live background services started for a run (rank 0 only)."""

    api_server: Optional[Any] = None
    dashboard: Optional[Any] = None


def _create_dashboard(cfg, run):
    """Create the terminal dashboard up front so the API server can log to it."""
    if not cfg.use_dashboard:
        return None
    try:
        from praxis.interface import TerminalDashboard

        # TerminalInterface starts it later; we only construct it here.
        return TerminalDashboard(cfg.seed, run.truncated_hash)
    except Exception as e:
        print(f"Warning: Could not create dashboard for API logging: {e}")
        return None


def _build_api_server(
    cfg, run, generator, tokenizer, integration_loader, param_stats, dashboard
):
    """Build the frontend, start the API server, and fire its hooks."""
    from praxis.environments import EnvironmentFeatures

    # Reload the web package so a dev iteration picks up recent changes.
    from praxis import web

    importlib.reload(web)
    from praxis.web import APIServer
    from praxis.web.src.build import build_dev

    print("[WEB] Building frontend...")
    build_dev()
    print("[WEB] ✓ Frontend build complete")

    api_server = APIServer(
        generator,
        cfg.host_name,
        cfg.port,
        tokenizer,
        integration_loader,
        param_stats,
        cfg.seed,
        truncated_hash=run.truncated_hash,
        full_hash=run.full_hash,
        dev_mode=(EnvironmentFeatures.get_active_environment() == "dev"),
        dashboard=dashboard,
        launch_command=run.full_command,
        config_file=getattr(cfg.args, "config_file", None),
    )
    api_server.start()

    for hook_func in integration_loader.get_api_server_hooks():
        hook_func(api_server.host, api_server.port)

    return api_server


def start_services(
    cfg, run, generator, tokenizer, integration_loader, param_stats, ckpt_path
):
    """Start rank-0 services and run integration init hooks (all ranks).

    Init hooks run before datasets are loaded so integrations are ready when
    their datasets are checked.
    """
    dashboard = None
    api_server = None
    if cfg.local_rank == 0:
        dashboard = _create_dashboard(cfg, run)
        api_server = _build_api_server(
            cfg, run, generator, tokenizer, integration_loader, param_stats, dashboard
        )

    integration_loader.run_init_hooks(
        cfg.args,
        run.cache_dir,
        ckpt_path=ckpt_path,
        truncated_hash=run.truncated_hash,
    )

    return Services(api_server=api_server, dashboard=dashboard)
