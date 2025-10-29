from pathlib import Path

from scaner.build_scan import create_scan_corpus


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('note_events', type=Path,
                        help='Path to NOTEEVENTS.csv from MIMIC-III.')
    parser.add_argument('outdir', type=Path,
                        help='Path to non-existent outdirectory to create ScAN.')
    parser.add_argument('-s', '--chunk-size', type=int, default=20,
                        help='Number of sentences in each chunk.')
    parser.add_argument('-o', '--overlap', type=int, default=5,
                        help='Number of overlapping sentences between chunks.')
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error(f'Chunk size must be positive: {args.chunk_size}')
    if args.overlap < 0:
        parser.error(f'Overlap must be non-negative: {args.overlap}')
    if args.overlap >= args.chunk_size:
        parser.error(f'Overlap ({args.overlap}) must be less than chunk size ({args.chunk_size})')

    create_scan_corpus(args.note_events, args.outdir, args.chunk_size, args.overlap)


if __name__ == '__main__':
    main()
