"""
Build ScAN, the suicide attempt-annotated MIMIC-III Dataset from NOTEEVENTS.csv using existing annotations.
"""
import csv
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from scaner.annotate_scan import write_annotations
from scaner.loader import load_labels, load_guidelines, load_json, load_pickle, load_annotations
from scaner.spacy_manager import get_sections

MIMIC_LINE_COUNT = 81_799_905


def get_guideline_value(guidelines, category, title, header):
    """
    Get the value from the guideline if it exists
    Args:
        guidelines:
        category:
        title:
        header:

    Returns: (int) - not sure what it signifies

    """
    try:
        return guidelines[category][title][header or None]
    except KeyError:
        return None


def text_from_span(doc, start, end):
    return doc[start:end].lower()


def iter_mimic_sections(guidelines_json, labels_pkl, noteevents_csv):
    guidelines = load_json(guidelines_json) if guidelines_json is not None else load_guidelines()
    sids, hadmids = load_pickle(labels_pkl) if labels_pkl is not None else load_labels()

    with open(noteevents_csv) as fh:
        for row in tqdm(csv.DictReader(fh), total=MIMIC_LINE_COUNT):
            # extract sections that may contain suicide-related info
            category = row['CATEGORY'].lower()
            sid = row['SUBJECT_ID']
            hadmid = row['HADM_ID']
            text = row['TEXT']
            if category not in guidelines:
                # WHY? no info on sections?
                continue
            if not sid or not hadmid:
                # WHY? perhaps missing annotations lookups?
                # are any of these '0'?
                continue
            elif int(sid) in sids and int(hadmid) in hadmids:
                result_text = []
                for title, header, section in get_sections(text):
                    if get_guideline_value(guidelines, category, title, header):
                        result_text.append(section)
                yield sid, hadmid, '\n\n'.join(result_text)


def corpus_text_to_files(noteevents_csv: Path, outdir: Path, labels_pkl: Path = None, guidelines_json: Path = None):
    i = 0
    for i, (sid, hadmid, text) in enumerate(
            iter_mimic_sections(guidelines_json, labels_pkl, noteevents_csv)
    ):
        with open(outdir / f'{sid}_{hadmid}', 'a') as out:
            out.write(text)
    logger.info(f'Wrote {i} sections to files in {outdir}.')


def create_scan_corpus(noteevents_csv: Path, outdir: Path, chunk_size=20, overlap=5,
                       labels_pkl: Path = None, guidelines_json: Path = None):
    """Create the ScAN corpus by applying annotations to the relevant sections in the MIMIC III corpus."""
    outdir.mkdir(exist_ok=True)
    files_dir = outdir / 'files'
    files_dir.mkdir(exist_ok=True)
    corpus_text_to_files(noteevents_csv, files_dir, labels_pkl, guidelines_json)
    for annot_prefix in {'train', 'test', 'val'}:
        annot_dict = load_annotations(annot_prefix)
        outfile = outdir / f'{annot_prefix}.jsonl'
        write_annotations(annot_dict, files_dir, outfile, chunk_size=chunk_size, overlap=overlap)
