"""Lab 4 starter: per-model Arabic normalisation profiles."""

import re
from dataclasses import dataclass

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer


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


# Lazily initialised on first use, since loading the disambiguator/
# morphology DB is slow and only needed when segment() is actually called.
_disambiguator = None
_tokenizer = None


def _get_tokenizer():
    global _disambiguator, _tokenizer

    if _tokenizer is None:
        _disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")
        _tokenizer = MorphologicalTokenizer(
            disambiguator=_disambiguator,
            scheme="atbtok",
            split=True,
        )

    return _tokenizer


def segment(text: str) -> list[str]:
    """Split Arabic text into clitic-segmented tokens using CAMeL Tools.

    Non-Arabic tokens (e.g. English words, numbers, IDs) are passed
    through unchanged, since the morphological analyser only segments
    Arabic script.
    """

    tokenizer = _get_tokenizer()

    words = text.split()
    segmented_tokens = []

    for word in words:
        if re.search(r"[\u0600-\u06FF]", word):
            pieces = tokenizer.tokenize([word])
            segmented_tokens.extend(pieces)
        else:
            segmented_tokens.append(word)

    # Strip clitic-boundary markers ("+") to match the plain-token
    # format used in the prepared bayan_ner_segmented.conll data.
    cleaned_tokens = [
        token.replace("+", "")
        for token in segmented_tokens
    ]

    return cleaned_tokens