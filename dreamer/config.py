import tomllib
from pathlib import Path
from types import SimpleNamespace

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.toml"


def load_config(section: str) -> SimpleNamespace:
    """Load config.toml, merge [common] + [section], return SimpleNamespace.

    section: 'sim' or 'real'
    Derived fields computed here: channels, observation_size.
    Device is set later by setup_device().
    """
    with open(_CONFIG_PATH, "rb") as f:
        raw = tomllib.load(f)

    if section not in raw:
        raise ValueError(f"Unknown config section '{section}'. Expected 'sim' or 'real'.")

    cfg = {**raw["common"], **raw[section]}

    cfg["channels"] = 1 if cfg["grayscale"] else 3
    cfg["observation_size"] = (cfg["channels"], 64, 64)

    return SimpleNamespace(**cfg)
