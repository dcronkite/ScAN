import json
import string
from enum import IntEnum
from pathlib import Path

from tqdm import tqdm

from scaner.spacy_manager import apply_nlp


def clean_string():
    keep = string.ascii_letters + string.digits + string.punctuation + ' '
    table = str.maketrans('', '', ''.join(c for c in map(chr, range(128)) if c not in keep))

    def _clean_string(text):
        text = ' '.join(text.split())
        text = text.translate(table)
        return text.strip()

    return _clean_string


def sentence_split(text):
    doc = apply_nlp(text)
    cleaner = clean_string()
    for sent in doc.sents:
        if text := cleaner(sent.text):
            yield {
                'text': text,
                'start': sent.start_char,
                'end': sent.end_char,
            }


def get_label(annotations):
    """
    Gets the label for both suicide attempt and suicide ideation.

    Args:
        annotations:

    Returns:

    """
    suicide_attempt = [(3, 'neutral')]
    suicide_ideation = [(3, 'neutral')]
    for annot in annotations:
        if sa := process_suicide_attempt(annot):
            suicide_attempt.append(sa)
        if si := process_suicide_ideation(annot):
            suicide_ideation.append(si)
    return {
        'suicide_attempt': sorted(suicide_attempt)[0][1],
        'suicide_ideation': sorted(suicide_ideation)[0][1],
        'relevance': 'pos' if len(suicide_attempt) > 1 or len(suicide_ideation) > 1 else 'neg',
    }


def process_suicide_ideation(annot):
    """suicide_ideation: labels: ["present", "neutral", "n/a", "absent"]"""
    if 'suicide_ideation' in annot:
        match annot['status']:
            case 'present':
                return 0, 'present'
            case 'N/A':
                return 2, 'n/a'
            case 'absent':
                return 1, 'absent'


def process_suicide_attempt(annot):
    """suicide_attempt: labels: ["neutral", "pos", "neg", "unsure"]"""
    if 'suicide_attempt' in annot:
        if all(annot[x] == 'N/A' for x in ['period', 'frequency', 'category']):
            return 1, 'neg'
        elif annot['category'] == 'unsure':
            return 2, 'unsure'
        else:
            return 0, 'pos'


def get_annotations(annot_dict, corpus_dir: Path, chunk_size=20, overlap=5, joiner=' '):
    """Read files, split into sentences, and add relevant annotations."""
    for key, annots in tqdm(annot_dict.items()):
        text = (corpus_dir / key).read_text(encoding='utf8')
        sentences = list(sentence_split(text))
        step = chunk_size - overlap
        n_chunks = ((len(sentences) - overlap) + (step - 1)) // step
        for i in range(n_chunks if n_chunks > 0 else 1):
            i *= step
            curr = sentences[i: i + chunk_size]
            start_idx = curr[0]['start']
            end_idx = curr[-1]['end']
            # collect annotations for this particular chunk
            curr_annotations = [annot for k, annot in annots.items()
                                if int(annot['annotation'][0]) <= end_idx
                                and int(annot['annotation'][1]) >= start_idx]
            curr_labels = get_label(curr_annotations)
            annotation = {
                'text': joiner.join(s['text'] for s in curr),
                'start': start_idx,
                'end': end_idx,
                'annotations': curr_annotations,
                'labels': curr_labels,
                'id': f'{key}_{i}',
            }
            label = curr_labels | {'text': annotation['text'], 'id': annotation['id']}
            yield annotation, label
