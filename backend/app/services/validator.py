"""Indian vehicle registration number validation and pattern matching."""

import re
from typing import Dict, Optional, Tuple

# Comprehensive Indian State & Union Territory 2-letter codes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "UT", "WB",
}

# Regex for standard format: SS DD LL NNNN or SS DD L NNNN or SS DD NNNN (where SS=State, DD=District, LL=Series, NNNN=Number)
STANDARD_PLATE_REGEX = re.compile(
    r"^([A-Z]{2})([0-9]{1,2})([A-Z]{0,3})([0-9]{4})$"
)

# Regex for BH Series (Bharat Series): YY BH NNNN LL (e.g. 22BH1234AA)
BH_SERIES_REGEX = re.compile(
    r"^([0-9]{2})(BH)([0-9]{4})([A-Z]{1,2})$"
)

# Regex for Vintage / Diplomatic / Defense / Commercial patterns
DIPLOMATIC_REGEX = re.compile(r"^([0-9]{1,3})(CD|CC|UN)([0-9]{1,4})$")
DEFENSE_REGEX = re.compile(r"^([0-9]{2}[A-Z][0-9]{5,6}[A-Z]?)$")


def validate_indian_plate_format(normalized_text: str) -> Tuple[str, Dict[str, Optional[str]]]:
    """Validates normalized registration string and returns (status, components).
    Status is one of: 'valid', 'possible', 'uncertain'.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", normalized_text.upper())
    if not cleaned:
        return "uncertain", {}

    # Check BH Series first (e.g. 22BH1234AB)
    bh_match = BH_SERIES_REGEX.match(cleaned)
    if bh_match:
        year, bh_code, num, series = bh_match.groups()
        return "valid", {
            "type": "BH_SERIES",
            "year": year,
            "code": bh_code,
            "number": num,
            "series": series,
            "formatted": f"{year} {bh_code} {num} {series}",
        }

    # Check Standard State Plate (e.g. GJ01AB1234, DL01CA1234, MH121234)
    std_match = STANDARD_PLATE_REGEX.match(cleaned)
    if std_match:
        state, district, series, num = std_match.groups()
        # Verify if state code is legitimate
        if state in INDIAN_STATE_CODES:
            formatted = f"{state} {district} {series} {num}" if series else f"{state} {district} {num}"
            return "valid", {
                "type": "STANDARD",
                "state": state,
                "district": district,
                "series": series or None,
                "number": num,
                "formatted": formatted.strip(),
            }
        else:
            # Structurally matches but state code unknown -> 'possible'
            formatted = f"{state} {district} {series} {num}" if series else f"{state} {district} {num}"
            return "possible", {
                "type": "STANDARD_UNKNOWN_STATE",
                "state": state,
                "district": district,
                "series": series or None,
                "number": num,
                "formatted": formatted.strip(),
            }

    # Check Diplomatic
    dip_match = DIPLOMATIC_REGEX.match(cleaned)
    if dip_match:
        country, code, num = dip_match.groups()
        return "valid", {
            "type": "DIPLOMATIC",
            "country": country,
            "code": code,
            "number": num,
            "formatted": f"{country} {code} {num}",
        }

    # Partial / Possible plate pattern heuristics
    # e.g., standard format with 1-3 digits at the end or 1 char state
    if len(cleaned) >= 6 and len(cleaned) <= 12:
        # Check if first 2 are alpha or close
        if cleaned[:2].isalpha() and any(c.isdigit() for c in cleaned[2:]):
            return "possible", {
                "type": "PARTIAL_MATCH",
                "raw": cleaned,
                "formatted": cleaned,
            }

    return "uncertain", {"raw": cleaned, "formatted": cleaned}
