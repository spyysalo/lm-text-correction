#!/usr/bin/env python3

import sys
import json

from collections import Counter
from argparse import ArgumentParser

from sentence_splitter import SentenceSplitter


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--language', default='fi')
    ap.add_argument('--min-chars', type=int, default=10)
    ap.add_argument('--max-chars', type=int, default=500)
    ap.add_argument('jsonl', nargs='+')
    return ap


def split_into_sentences(text, args):
    lines = [l for l in text.split('\n')]
    lines = [l for l in lines if l and not l.isspace()]

    splitter = SentenceSplitter(language=args.language)
    sentences = [s for l in lines for s in splitter.split(l)]
    sentences = [s for s in sentences if s and not s.isspace()]

    return sentences


def keep_sentence(sentence, args):
    if len(sentence) < args.min_chars:
        return False
    elif len(sentence) > args.max_chars:
        return False
    else:
        return True


def main(argv):
    args = argparser().parse_args()

    total_count, output_count = 0, 0
    for fn in args.jsonl:
        with open(fn) as f:
            for l in f:
                data = json.loads(l)
                text = data['text']
                sentences = split_into_sentences(text, args)
                total_count += len(sentences)
                sentences = [s for s in sentences if keep_sentence(s, args)]
                for s in sentences:
                    print(json.dumps({ 'text': s }, ensure_ascii=False))
                output_count += len(sentences)

    print(f'DONE, output {output_count}/{total_count} sentences',
          f'({output_count/total_count:.1%})',
          file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
