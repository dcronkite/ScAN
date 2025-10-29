"""
Simplify interface to getting spacy, and reduce loads.
"""
import medspacy

_SPACY_NLP = None


def get_spacy():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        _SPACY_NLP = medspacy.load(enable=['section_detection'])
        _SPACY_NLP.add_pipe('medspacy_sectionizer')
        _SPACY_NLP.max_length = 1_500_000
    return _SPACY_NLP


def apply_nlp(text):
    return get_spacy()(text)


def get_sections(text):
    doc = apply_nlp(text)
    for section in doc._.sections:
        header = doc[section.title_start: section.title_end].text
        section_text = doc[section.title_start: section.body_end].text
        yield (section.category.lower() if section.category is not None else None,
               # 'title' (i.e., normalized name of section)
               header.lower(),  # just header
               section_text.lower(),  # header + body
               )
