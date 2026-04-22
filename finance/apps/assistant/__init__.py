"""
finance.apps.assistant
======================
Trading Assistant — evening prep workflow.

Scored watchlist, market regime panel, AI analysis, and ticker export.
Replaces finance.apps.analyst and finance.apps.conditions.
"""
from __future__ import annotations

APP_NAME = "assistant"
APP_DESCRIPTION = "Trading Assistant — scored watchlist and market context"
APP_ICON_ID = "assistant"


def launch(*, archive: bool = False, dry_run: bool = False, **_kwargs) -> None:
    """
    Launch the Trading Assistant.

    Parameters
    ----------
    archive:
        If True, run the headless archive pipeline (no Qt) and write today's
        Parquet file to finance/_data/assistant/YYYY-MM-DD.parquet.
    dry_run:
        If True, skip writing the archive file (archive mode only).
    """
    if archive:
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
        from finance.apps.assistant._runner import run_archive
        path = run_archive(dry_run=dry_run)
        if dry_run:
            print(f"Archive DRY RUN — would write: {path}")
        else:
            print(f"Archive written: {path}")
        return

    from finance.apps._qt_bootstrap import (
        apply_dark_palette,
        ensure_ipython_event_loop,
        ensure_qt_app,
        exec_or_return,
    )
    from finance.apps.assistant._window import AssistantWindow

    ensure_ipython_event_loop()
    app = ensure_qt_app()
    apply_dark_palette(app)

    win = AssistantWindow()
    win.show()
    exec_or_return(app)
