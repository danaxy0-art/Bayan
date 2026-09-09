"""Lab 4 starter: per-model Arabic normalisation profiles."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


_DIACRITICS_RE = re.compile(r"[\u064B-\u0652\u0670]")

_TATWEEL = "\u0640"

_ALEF_FORMS_RE = re.compile(r"[إأآٱ]")

_ALEF_MAKSURA = "ى"
_YAA = "ي"

_TEH_MARBUTA = "ة"
_HAA = "ه"

_WAW_HAMZA = "ؤ"
_WAW = "و"

_YAA_HAMZA = "ئ"


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Normalise Arabic text according to the given profile.

    Applies, in order:
      1. Tatweel/kashida removal
      2. Diacritic (tashkeel) removal, only if profile.dediacritize
      3. Alef unification (أ إ آ ٱ -> ا)
      4. Alef maksura -> yaa (ى -> ي)
      5. Teh marbuta -> haa (ة -> ه)
      6. Hamza-on-waw/yaa -> bare waw/yaa (ؤ -> و, ئ -> ي)
    """

    normalized = text

    normalized = normalized.replace(_TATWEEL, "")

    if profile.dediacritize:
        normalized = _DIACRITICS_RE.sub("", normalized)

    normalized = _ALEF_FORMS_RE.sub("ا", normalized)

    normalized = normalized.replace(_ALEF_MAKSURA, _YAA)

    normalized = normalized.replace(_TEH_MARBUTA, _HAA)

    normalized = normalized.replace(_WAW_HAMZA, _WAW)
    normalized = normalized.replace(_YAA_HAMZA, _YAA)

    return normalized


def segment(text: str) -> list[str]:
    # TODO(Lab 4 - Step 3): wire the chosen CAMeL Tools clitic segmentation scheme.
    raise NotImplementedError