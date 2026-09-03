"""Position-aware Indian vehicle registration OCR text normalizer, multi-line aggregator and error corrector.
Includes intelligent O vs D disambiguation for RTO series codes and format classification.
"""

import re
from typing import List, Tuple
from app.services.validator import (
    INDIAN_STATE_CODES,
    validate_indian_plate_format,
)

# Character confusion mappings
CHAR_TO_DIGIT = {
    "O": "0", "D": "0", "Q": "0",
    "I": "1", "L": "1", "T": "1", "|": "1", "J": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6", "C": "0",
    "B": "8",
}

DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}


def clean_raw_ocr(raw_text: str) -> str:
    """Removes noise, punctuation, hyphens, screws, dots and whitespace, converting to uppercase."""
    if not raw_text:
        return ""
    cleaned = raw_text.upper()
    # Strip common HSRP country badges / stamps (IND, INDIA, BHARAT)
    cleaned = re.sub(r"\b(IND|INDIA|BHARAT)\b", "", cleaned)
    # Remove leading IND if glued to text
    cleaned = re.sub(r"^(IND|IN)", "", cleaned)
    # Remove non-alphanumeric
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    return cleaned


def normalize_indian_plate(raw_text: str) -> Tuple[str, str, str]:
    """Normalizes raw OCR text using position-aware heuristics for Indian registration plates.
    Disambiguates 'O' vs 'D' in series letter positions (e.g. 'OS' -> 'DS', 'OA' -> 'DA').

    Returns:
        (normalized_text, formatted_text, format_status)
    """
    cleaned = clean_raw_ocr(raw_text)
    if not cleaned:
        return "", "", "uncertain"

    # 1. Direct validation check before aggressive correction
    status, comp = validate_indian_plate_format(cleaned)
    if status == "valid":
        # Check if 'O' is in series position where 'D' is standard (e.g. GJ27OS4837 -> GJ27DS4837)
        if len(cleaned) == 10 and cleaned[4] == "O" and cleaned[5] in ["S", "A", "E", "B", "C", "D", "P", "R", "T", "V", "N", "M", "K"]:
            candidate_d = cleaned[:4] + "D" + cleaned[5:]
            st_d, comp_d = validate_indian_plate_format(candidate_d)
            if st_d == "valid":
                return candidate_d, comp_d.get("formatted", candidate_d), "valid"

        return cleaned, comp.get("formatted", cleaned), "valid"

    chars = list(cleaned)
    n = len(chars)

    # 2. BH Series check: Chars 2-3 are 'BH' or '8H'
    if n >= 9 and n <= 11:
        if chars[2] in ["B", "8"] and chars[3] in ["H", "4"]:
            bh_chars = list(chars)
            bh_chars[0] = CHAR_TO_DIGIT.get(bh_chars[0], bh_chars[0])
            bh_chars[1] = CHAR_TO_DIGIT.get(bh_chars[1], bh_chars[1])
            bh_chars[2] = "B"
            bh_chars[3] = "H"
            for i in range(4, min(8, n)):
                bh_chars[i] = CHAR_TO_DIGIT.get(bh_chars[i], bh_chars[i])
            for i in range(8, n):
                bh_chars[i] = DIGIT_TO_CHAR.get(bh_chars[i], bh_chars[i])

            bh_candidate = "".join(bh_chars)
            bh_status, bh_comp = validate_indian_plate_format(bh_candidate)
            if bh_status == "valid":
                return bh_candidate, bh_comp.get("formatted", bh_candidate), "valid"

    # 3. Standard State Plate correction (e.g., GJ 01 AB 1234, GJ 27 DS 4837)
    if n in [8, 9, 10, 11]:
        std_chars = list(chars)
        # Position 0-1: State Code (Letters)
        std_chars[0] = DIGIT_TO_CHAR.get(std_chars[0], std_chars[0])
        std_chars[1] = DIGIT_TO_CHAR.get(std_chars[1], std_chars[1])

        # If 10 chars: SS (0,1) DD (2,3) LL (4,5) NNNN (6,7,8,9)
        if n == 10:
            std_chars[2] = CHAR_TO_DIGIT.get(std_chars[2], std_chars[2])
            std_chars[3] = CHAR_TO_DIGIT.get(std_chars[3], std_chars[3])
            
            # Position 4 & 5 (Series Letters):
            # '0' and 'O' are frequently misread for 'D' in RTO series
            c4 = std_chars[4]
            c5 = std_chars[5]
            if c4 in ["0", "O", "Q"]:
                std_chars[4] = "D"
            else:
                std_chars[4] = DIGIT_TO_CHAR.get(c4, c4)

            if c5 in ["0", "O", "Q"] and std_chars[4] != "D":
                std_chars[5] = "D"
            else:
                std_chars[5] = DIGIT_TO_CHAR.get(c5, c5)

            std_chars[6] = CHAR_TO_DIGIT.get(std_chars[6], std_chars[6])
            std_chars[7] = CHAR_TO_DIGIT.get(std_chars[7], std_chars[7])
            std_chars[8] = CHAR_TO_DIGIT.get(std_chars[8], std_chars[8])
            std_chars[9] = CHAR_TO_DIGIT.get(std_chars[9], std_chars[9])
        elif n == 9:
            # SS DD L NNNN
            std_chars[2] = CHAR_TO_DIGIT.get(std_chars[2], std_chars[2])
            std_chars[3] = CHAR_TO_DIGIT.get(std_chars[3], std_chars[3])
            c4 = std_chars[4]
            std_chars[4] = "D" if c4 in ["0", "O", "Q"] else DIGIT_TO_CHAR.get(c4, c4)
            for i in range(5, 9):
                std_chars[i] = CHAR_TO_DIGIT.get(std_chars[i], std_chars[i])
        elif n == 8:
            # SS DD NNNN
            std_chars[2] = CHAR_TO_DIGIT.get(std_chars[2], std_chars[2])
            std_chars[3] = CHAR_TO_DIGIT.get(std_chars[3], std_chars[3])
            for i in range(4, 8):
                std_chars[i] = CHAR_TO_DIGIT.get(std_chars[i], std_chars[i])
        elif n == 11:
            # SS DD LLL NNNN
            std_chars[2] = CHAR_TO_DIGIT.get(std_chars[2], std_chars[2])
            std_chars[3] = CHAR_TO_DIGIT.get(std_chars[3], std_chars[3])
            std_chars[4] = "D" if std_chars[4] in ["0", "O", "Q"] else DIGIT_TO_CHAR.get(std_chars[4], std_chars[4])
            std_chars[5] = DIGIT_TO_CHAR.get(std_chars[5], std_chars[5])
            std_chars[6] = DIGIT_TO_CHAR.get(std_chars[6], std_chars[6])
            for i in range(7, 11):
                std_chars[i] = CHAR_TO_DIGIT.get(std_chars[i], std_chars[i])

        candidate = "".join(std_chars)
        cand_status, cand_comp = validate_indian_plate_format(candidate)
        if cand_status == "valid":
            return candidate, cand_comp.get("formatted", candidate), "valid"
        elif cand_status == "possible":
            return candidate, cand_comp.get("formatted", candidate), "possible"

    final_status, final_comp = validate_indian_plate_format(cleaned)
    formatted = final_comp.get("formatted", cleaned)
    return cleaned, formatted, final_status
