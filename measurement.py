"""Optional CodeCarbon measurements with an explicit estimate fallback.

CodeCarbon estimates energy use and CO2-equivalent emissions from available hardware
telemetry. These are not physical meter readings. The fallback uses the project's
configured power-times-runtime assumptions and is labelled separately in results.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import re
import time
from pathlib import Path
from typing import Any, Callable

from datadiet.measure import Assumptions, account


@dataclass
class Measurement:
    values: dict[str, float]
    source: str
    seconds: float


def measured_account(
    seconds: float,
    assumptions: Assumptions | None = None,
    energy_kwh: float | None = None,
    emissions_kg: float | None = None,
) -> dict[str, float]:
    """Return kwh, gco2, rupees, using CodeCarbon values when both are provided."""
    assumptions = assumptions or Assumptions()
    if energy_kwh is None or emissions_kg is None:
        return account(seconds, assumptions)
    return {
        "kwh": max(0.0, float(energy_kwh)),
        "gco2": max(0.0, float(emissions_kg)) * 1000.0,
        # CodeCarbon does not measure monetary cost; this remains a configured estimate.
        "rupees": assumptions.rupees_per_hour * max(0.0, float(seconds)) / 3600.0,
    }


def _read_last_report_row(report_path: Path) -> dict[str, str]:
    """Read the last non-empty row from a CodeCarbon CSV report, if available."""
    if not report_path.exists():
        return {}
    try:
        with report_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return rows[-1] if rows else {}
    except (OSError, csv.Error):
        return {}


def measure_callable(
    fn: Callable[[], Any],
    *,
    assumptions: Assumptions | None = None,
    project_name: str = "datadiet",
    output_dir: str = "benchmarks/codecarbon",
    country_iso_code: str = "IND",
    run_id: str = "run",
    allow_fallback: bool = True,
) -> tuple[Any, Measurement]:
    """Run ``fn`` under CodeCarbon and return its result plus impact measurements.

    ``source`` is ``"codecarbon"`` only when both energy and emissions are obtained
    from CodeCarbon. If tracking cannot start or complete, the runtime-based estimate
    is used only when ``allow_fallback=True``. With ``False``, tracking errors raise.
    Exceptions raised by ``fn`` are propagated after the tracker is stopped.
    """
    assumptions = assumptions or Assumptions()
    output_path = Path(output_dir)
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("._") or "run"
    # Unique report names avoid colliding with previous runs or different strategies.
    report_name = f"{safe_id}_{time.time_ns()}.csv"
    report_path = output_path / report_name
    tracker = None

    try:
        from codecarbon import OfflineEmissionsTracker

        output_path.mkdir(parents=True, exist_ok=True)
        tracker = OfflineEmissionsTracker(
            project_name=project_name,
            output_dir=str(output_path),
            output_file=report_name,
            country_iso_code=country_iso_code,
            log_level="error",
            save_to_file=True,
            measure_power_secs=1,
        )
        tracker.start()
    except Exception as exc:
        if tracker is not None:
            try:
                tracker.stop()
            except Exception:
                pass
        if not allow_fallback:
            raise RuntimeError(
                "CodeCarbon could not start. Install/repair CodeCarbon or omit "
                "--require-codecarbon to permit labelled estimate fallback."
            ) from exc
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        return result, Measurement(account(elapsed, assumptions), "estimate_fallback", elapsed)

    start = time.perf_counter()
    try:
        result = fn()
    except BaseException:
        try:
            tracker.stop()
        except Exception:
            pass
        raise

    elapsed = time.perf_counter() - start
    try:
        emissions_kg = tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        energy_kwh = getattr(data, "energy_consumed", None) if data is not None else None
        row = _read_last_report_row(report_path)
        if energy_kwh is None and row.get("energy_consumed") not in (None, ""):
            energy_kwh = float(row["energy_consumed"])
        if emissions_kg is None and row.get("emissions") not in (None, ""):
            emissions_kg = float(row["emissions"])
        if energy_kwh is None or emissions_kg is None:
            raise RuntimeError("CodeCarbon did not provide both energy and emissions")
        values = measured_account(elapsed, assumptions, energy_kwh, emissions_kg)
        return result, Measurement(values, "codecarbon", elapsed)
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError(
                "CodeCarbon ran but did not return usable energy and emissions data."
            ) from exc
        return result, Measurement(account(elapsed, assumptions), "estimate_fallback", elapsed)
