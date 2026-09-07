"""Lab 1 starter: sentence segmentation."""

import spacy

from bayan.preprocessing.core import preprocess

def build_pipeline():
        #blank multilingual spacy pipline
    nlp = spacy.blank("xx")

    #add lightweight sentence segmentation component.
    nlp.add_pipe("sentencizer", config=config)

    return nlp
    raise NotImplementedError("Implement build_pipeline() in Lab 1")


def split_sentences(raw: str, nlp) -> list[str]:
   # Important: use the same preprocessing contract first.
text = preprocess (raw)
doc = ntp(text)

return [
    sent.text.strip()
    for sent in doc.sents 
    if sent.text.strip()
]
    raise NotImplementedError("Implement split_sentences() in Lab 1")

"""
test it : python
from bayan preprocessing. segmentation import build_pipeline, split_sentences
nip = build_pipeline()
". الخدمة معتازا. لكن التطبيل بطرء. أرجو حل المشكلة " = text
print (split_sentences(text, ntp))
".١ رقمى 0551234567. الخدمة ممنازة" = text
print(split_sentences(text, nlp))
print()
  """  
