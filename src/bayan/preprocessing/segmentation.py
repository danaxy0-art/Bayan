"""Lab 1 starter: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess


def build_pipeline():
    # Blank multilingual spaCy pipeline.
    nlp = spacy.blank("xx")

    # Add lightweight sentence segmentation component.
    nlp.add_pipe("sentencizer") #set end & start of sentence by . ? !

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    # Important: use the same preprocessing contract first.
    text = preprocess(raw)

    doc = nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]
    
    
"""
test it using : 
in cmd write:
1-
python

2-
from bayan.preprocessing.segmentation import build_pipeline, split_sentences
nlp = build_pipeline()
text = "الخدمة ممتازة. لكن التطبيق بطيء. أرجو حل المشكلة."
print(split_sentences(text, nlp))
text = "رقمي 0551234567. الخدمــــة ممتازة 😍."
print(split_sentences(text, nlp))

3-
exit()
"""
