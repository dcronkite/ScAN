from pathlib import Path

from scaner.build_scan import create_scan_corpus


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('note_events', type=Path,
                        help='Path to NOTEEVENTS.csv from MIMIC-III.')
    parser.add_argument('outdir', type=Path,
                        help='Path to non-existent outdirectory to create ScAN.')
    args = parser.parse_args()
    create_scan_corpus(args.note_events, args.outdir)


if __name__ == '__main__':
    main()
