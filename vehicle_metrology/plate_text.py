"""Visual Latin-to-Cyrillic normalization for Russian registration numbers."""
import re

TRANSLATION = str.maketrans('ABCEHKMOPTXY', 'АВСЕНКМОРТХУ')


def normalize_plate(text):
    return re.sub(r'\s+', '', text.upper().translate(TRANSLATION))
