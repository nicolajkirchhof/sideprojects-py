"""
finance.apps
=============
Central registry for all finance applications.

Usage::

    python -m finance.apps                  # list available apps
    python -m finance.apps swing-plot       # launch swing plot dashboard
    python -m finance.apps momentum         # launch momentum dashboard
"""
from __future__ import annotations

import importlib
from typing import Any


APPS: dict[str, str] = {
    "swing-plot":  "finance.apps.swing_plot",
    "momentum":    "finance.apps.momentum_dashboard",
    "conditions":  "finance.apps.conditions",
    "analyst":     "finance.apps.analyst",
}


def list_apps() -> dict[str, str]:
    """Return {name: description} for all registered apps."""
    result = {}
    for name, module_path in APPS.items():
        mod = importlib.import_module(module_path)
        result[name] = getattr(mod, "APP_DESCRIPTION", "(no description)")
    return result


def get_app(name: str) -> Any:
    """Import and return the app module by its registered name."""
    if name not in APPS:
        raise KeyError(
            f"Unknown app '{name}'. Available: {', '.join(sorted(APPS))}"
        )
    return importlib.import_module(APPS[name])
