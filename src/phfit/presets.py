"""Preset pKa values for common fluorescent pH indicator reagents."""

PRESETS = {
    "oregongreen488": {
        "name": "Oregon Green 488",
        "pKa": 4.7,
        "default_n": 1,
        "description": "Oregon Green 488 BAPTA, FITC derivative",
    },
    "oregongreen514": {
        "name": "Oregon Green 514",
        "pKa": 4.8,
        "default_n": 1,
        "description": "Oregon Green 514, red-shifted variant",
    },
    "phrodo_red": {
        "name": "pHrodo Red",
        "pKa": 6.5,
        "default_n": -1,
        "description": "pHrodo Red, increases fluorescence at low pH",
    },
    "phrodo_green": {
        "name": "pHrodo Green",
        "pKa": 6.5,
        "default_n": -1,
        "description": "pHrodo Green, increases fluorescence at low pH",
    },
    "lysosensor_green": {
        "name": "LysoSensor Green DND-189",
        "pKa": 5.2,
        "default_n": -1,
        "description": "LysoSensor Green DND-189, accumulates in acidic organelles",
    },
    "lysosensor_blue": {
        "name": "LysoSensor Blue DND-167",
        "pKa": 5.1,
        "default_n": -1,
        "description": "LysoSensor Blue DND-167",
    },
    "lysosensor_yellowblue": {
        "name": "LysoSensor Yellow/Blue DND-160",
        "pKa": 4.2,
        "default_n": -1,
        "description": "LysoSensor Yellow/Blue DND-160, dual-emission ratiometric",
    },
    "bcecf": {
        "name": "BCECF",
        "pKa": 6.98,
        "default_n": 1,
        "description": "BCECF, ratiometric (Ex 440/490), gold standard for cytoplasmic pH",
    },
    "fitc": {
        "name": "FITC / Fluorescein",
        "pKa": 6.4,
        "default_n": 1,
        "description": "Fluorescein isothiocyanate",
    },
    "snarf1": {
        "name": "SNARF-1",
        "pKa": 7.5,
        "default_n": 1,
        "description": "SNARF-1, ratiometric (Em 580/640)",
    },
    "cypher5e": {
        "name": "CypHer5E",
        "pKa": 7.3,
        "default_n": -1,
        "description": "CypHer5E, far-red, increases fluorescence at low pH",
    },
    "hpts": {
        "name": "HPTS (Pyranine)",
        "pKa": 7.3,
        "default_n": 1,
        "description": "HPTS, ratiometric (Ex 405/450), water-soluble",
    },
}


def list_presets() -> str:
    """Return a formatted string listing all available presets."""
    lines = []
    lines.append(f"{'Preset name':<25} {'Reagent':<35} {'pKa':>5}  {'Direction'}")
    lines.append("-" * 80)
    for key, info in PRESETS.items():
        direction = "\u2191 ascending" if info["default_n"] > 0 else "\u2193 descending"
        lines.append(f"{key:<25} {info['name']:<35} {info['pKa']:>5.2f}  {direction}")
    return "\n".join(lines)


def get_preset(name: str) -> dict:
    """Get preset info by name. Raises ValueError if not found."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(
            f"Unknown preset '{name}'. Available presets: {available}\n"
            f"Use 'phfit --preset list' to see all presets."
        )
    return PRESETS[key]
