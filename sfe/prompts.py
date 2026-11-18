"""Rule-based "ground" prompts for views that look down.

A view that only sees the ground must not be asked for sky content ("at sunset with an orange sky"),
or the model paints the sun onto the ground. ground_prompt() keeps the lighting implied by the
phrase and deletes the sky itself. Rules are applied in order (longer phrases first).
"""
import re

RULES = [
    (r"\bat sunset with an orange sky\b", "in warm orange evening light"),
    (r"\bat sunset\b", "in warm orange evening light"),
    (r"\bat sunrise\b", "in warm morning light"),
    (r"\bat dusk\b", "in dim dusk light"),
    (r"\bat night under a starry sky\b", "at night"),
    (r"\bunder a starry sky\b", ""),
    (r"\bon Mars under a red sky\b", "on Mars"),
    (r"\bunder (?:a|an) (?:cloudy|blue|sunny blue|sunny|overcast|red|clear|orange|grey|gray) sky\b", ""),
    (r"\bwith an orange sky\b", ""),
    (r"\bon a sunny day\b", "in bright sunlight"),
]


# "neutral": same deletions, but no light colour (a down view told "warm orange light" paints orange blobs)
NEUTRAL = {
    "in warm orange evening light": "in the evening",
    "in warm morning light": "in the morning",
    "in dim dusk light": "at dusk",
    "in bright sunlight": "on a sunny day",
}


def ground_prompt(text: str, style: str = "warm") -> str:
    if style not in ("warm", "neutral"):
        raise ValueError(style)
    out = text
    for pat, rep in RULES:
        if style == "neutral":
            rep = NEUTRAL.get(rep, rep)
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+,", ",", out)
    out = re.sub(r",\s*,", ",", out)
    return out.strip().rstrip(",").strip()


if __name__ == "__main__":
    import sys

    import yaml

    cfg = yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/sky8.yaml", encoding="utf-8"))
    for p in cfg["panoramas"]:
        for t in [p["src"]] + [e["tar"] for e in p["edits"]]:
            g = ground_prompt(t)
            print(("   " if g == t else "-> ") + t + ("" if g == t else "\n   => " + g))
