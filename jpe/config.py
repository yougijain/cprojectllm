from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_FILE = ROOT / "configs" / "models.yaml"


def load_models(path: Path = MODELS_FILE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def model_config(name: str, path: Path = MODELS_FILE) -> dict:
    cfg = load_models(path)
    if name == "teacher":
        return dict(cfg["teacher"], label="teacher:" + cfg["teacher"]["model"])
    try:
        entry = dict(cfg["models"][name])
    except KeyError:
        raise KeyError(f"no model {name!r} in {path}; have {', '.join(cfg['models'])}") from None
    entry.setdefault("label", name)
    return entry
