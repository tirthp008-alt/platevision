import re

INDIAN = re.compile(r"^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$")

def normalize_plate(raw: str) -> tuple[str, str, str]:
    value = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    # Only apply position-safe replacements after letters and digits are separated by a plausible pattern.
    value = re.sub(r"^([A-Z]{2})O", r"\g<1>0", value)
    value = re.sub(r"(\d)[OQI](?=\d)", lambda m: m.group(1) + {"O":"0","Q":"0","I":"1"}[m.group(0)[1]], value)
    status = "valid" if INDIAN.fullmatch(value) else ("possible" if len(value) >= 6 else "uncertain")
    match = re.match(r"^([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{4})$", value)
    formatted = " ".join(match.groups()) if match else value
    return value, formatted, status
