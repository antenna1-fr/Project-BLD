# src/features/data_preparer.py
"""
Thin wrapper around the existing data_preparer logic.

Goal: expose a clean function-level API that other code (models, backtests, RL)
can call, instead of everything living in a monolithic script.
"""

from pathlib import Path
from typing import Optional
import sys

# Add project root to path if not already there
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import PROCESSED_CSV, RAW_DB_PATH

# Import the main function from the refactored script module
# We import with a try/except to handle both during transition and after
try:
    from scripts.data_preparer import main as run_full_pipeline_impl
except ImportError:
    # Fallback during transition
    run_full_pipeline_impl = None


def build_full_processed_dataset(
    raw_db: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Path:
    """
    Run the two-pass feature pipeline and return the path to the processed CSV.

    This is the canonical entry point for everything else in the repo.

    Args:
        raw_db: Path to the raw SQLite database (defaults to config.RAW_DB_PATH)
        output_csv: Path for the output processed CSV (defaults to config.PROCESSED_CSV)

    Returns:
        Path to the processed CSV file
    """
    raw_db = Path(raw_db or RAW_DB_PATH)
    output_csv = Path(output_csv or PROCESSED_CSV)

    # For now, we call the main() function from scripts/data_preparer.py
    # The script uses global config values, so we need to update them temporarily
    if run_full_pipeline_impl is None:
        raise RuntimeError(
            "scripts.data_preparer module not found. "
            "Ensure scripts/data_preparer.py exists and is importable."
        )

    # Update config temporarily (the script reads from config module)
    import config as cfg
    old_db = cfg.DB_PATH
    old_csv = cfg.PROCESSED_CSV

    try:
        cfg.DB_PATH = raw_db
        cfg.PROCESSED_CSV = output_csv

        # Import here to get updated config
        from scripts import data_preparer
        data_preparer.DB_PATH = raw_db
        data_preparer.OUTPUT_CSV = output_csv

        # Run the pipeline
        run_full_pipeline_impl()

    finally:
        # Restore config
        cfg.DB_PATH = old_db
        cfg.PROCESSED_CSV = old_csv

    return output_csv


def run_full_pipeline(raw_db: Path | str = None, output_csv: Path | str = None) -> None:
    """
    Orchestrates PASS-1, scaler fit, PASS-2, and final concatenation.
    This is factored out so the build_full_processed_dataset can call it.

    Args:
        raw_db: Path to raw database (defaults to config value)
        output_csv: Path for output CSV (defaults to config value)
    """
    build_full_processed_dataset(
        raw_db=Path(raw_db) if raw_db else None,
        output_csv=Path(output_csv) if output_csv else None
    )


# Re-export key functions for convenience
__all__ = ['build_full_processed_dataset', 'run_full_pipeline']

