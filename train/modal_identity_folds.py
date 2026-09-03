"""Launch paired-particle identity-holdout jobs on ephemeral Modal A100s."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from identity_fold_payloads import build_payloads
from modal_train import app, run_experiment


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.local_entrypoint()
def main(mode: str = "pilot", record_path: str = "") -> None:
    if mode not in {"pilot", "full"}:
        raise SystemExit("--mode must be pilot or full")
    pilot = mode == "pilot"
    payloads = build_payloads(pilot=pilot)
    default_record = (
        "artifacts/identity_holdout/compute_plan.json"
        if pilot
        else "artifacts/identity_holdout/full_run_calls.json"
    )
    destination = Path(record_path or default_record)
    destination.parent.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    start = time.monotonic()
    calls = []
    for payload in payloads:
        call = run_experiment.spawn(json.dumps(payload, sort_keys=True))
        calls.append(
            {
                "tag": payload["cfg"]["tag"],
                "model_family": payload["cfg"]["model_family"],
                "heldout_particle": payload["cfg"]["heldout_particle"],
                "call_id": call.object_id,
                "call": call,
            }
        )

    completed = []
    failure = None
    try:
        for item in calls:
            result = item["call"].get()
            completed.append(
                {
                    key: value for key, value in item.items() if key != "call"
                }
                | {"status": "complete", "result": result}
            )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record = {
            "mode": mode,
            "gpu": "A100-40GB",
            "app": "gra29-sp-emulator",
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "wall_seconds": time.monotonic() - start,
            "requested_jobs": len(payloads),
            "completed_jobs": completed,
            "failure": failure,
            "pilot_metrics_are_manuscript_evidence": False,
            "payloads": payloads,
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {destination}", flush=True)
