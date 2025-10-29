import json
import string
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


def write_annotations(annot_dict, corpus_dir: Path, outfile: Path, chunk_size=20, overlap=5, joiner=' '):
    """Read files, split into sentences, and add relevant annotations."""
    with open(outfile, 'w', encoding='utf8') as out:
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
                out.write(
                    json.dumps(
                        {key: {
                            'text': joiner.join(s['text'] for s in curr),
                            'start': start_idx,
                            'end': end_idx,
                            'annotations': curr_annotations,
                        }}
                    ) + '\n')
