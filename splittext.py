#!/usr/bin/env python3

# Split "text" in JSONL into blocks with a given maximum number of
# tokens.

import sys
import json
import random
import re

from argparse import ArgumentParser


# Whitespace tokenization, keep whitespace
TOKENIZE_REGEX = re.compile(r'\S+|\s+')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--uniq', action='store_true')
    ap.add_argument('--normalize-space', action='store_true')
    ap.add_argument('--min-tokens', type=int, default=None)
    ap.add_argument('--short-prob', type=float, default=None)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('jsonl', nargs='+')
    return ap


def split_line(text, args):
    tokens = TOKENIZE_REGEX.findall(text)

    if len([t for t in tokens if not t.isspace()]) < args.max_tokens:
        yield text    # no need to split
    else:
        nonspace, block = 0, []
        for i, t in enumerate(tokens):
            block.append(t)
            if not tokens[i].isspace():
                nonspace += 1
            if nonspace >= args.max_tokens:
                yield ''.join(block).strip()
                nonspace, block = 0, []
        if block:
            yield ''.join(block).strip()


def split_text(text, args):
    lines = text.splitlines()
    lines = [l for l in lines if l and not l.isspace()]
    lines = [l.strip() for l in lines]

    blocks = []
    for line in lines:
        blocks.extend(split_line(line, args))

    if args.uniq:
        uniq = []
        for b in blocks:
            if b not in split_text.seen:
                uniq.append(b)
                split_text.seen.add(b)
        blocks = uniq

    if args.min_tokens is not None:
        filtered = []
        for b in blocks:
            if len(b.split()) >= args.min_tokens:
                filtered.append(b)
            elif (args.short_prob is not None and
                  random.random() < args.short_prob):
                filtered.append(b)    # keep short
        blocks = filtered

    if args.normalize_space:
        blocks = [' '.join(b.split()) for b in blocks]
        
    return blocks        
split_text.seen = set()


def main(argv):
    args = argparser().parse_args()

    random.seed(42)
    
    for fn in args.jsonl:
        with open(fn) as f:
            for l in f:
                data = json.loads(l)
                blocks = split_text(data['text'], args)
                for i, b in enumerate(blocks):
                    block = {
                        'id': f'{data["id"]}-{i}',
                        'text': b,
                    }
                    print(json.dumps(block, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
