#!/usr/bin/env python3

import sys
import json

import numpy as np

from string import ascii_letters, digits, punctuation
from argparse import ArgumentParser


# Default set of characters that can be substituted for
DEFAULT_CHARSET = ascii_letters + digits + punctuation + ' '


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mean', type=float, default=0.1)
    ap.add_argument('--stdev', type=float, default=0.1)
    ap.add_argument('--min-error-prob', type=float, default=0.0)
    ap.add_argument('--max-error-prob', type=float, default=0.5)
    ap.add_argument('--delete-prob', type=float, default=0.1)
    ap.add_argument('--insert-prob', type=float, default=0.1)
    ap.add_argument('--charset', default=DEFAULT_CHARSET)
    ap.add_argument('jsonl', nargs='+')
    return ap


def add_noise(text, rng, args):
    # draw probability of error
    prob = rng.normal(args.mean, args.stdev)
    prob = min(max(prob, args.min_error_prob), args.max_error_prob)

    chars = []
    for c in text:
        if rng.random() > prob:
            chars.append(c)    # no error
        elif rng.random() < args.delete_prob:
            pass    # delete
        elif rng.random() < args.insert_prob:
            chars.append(rng.choice(args.charset))    # insert
            chars.append(c)
        else:
            chars.append(rng.choice(args.charset))    # substitute

    return ''.join(chars)


def main(argv):
    args = argparser().parse_args()
    args.charset = [c for c in args.charset]    # for np.random.choice

    # Adjust inser_prob: as the possibility of insertion is only
    # considered if the deletion probability threshold is not
    # exceeded, it needs to be adjusted to match the CLI probability
    args.insert_prob = args.insert_prob / (1-args.delete_prob)

    rng = np.random.default_rng(args.seed)

    for fn in args.jsonl:
        with open(fn) as f:
            for line in f:
                indata = json.loads(line)
                text = indata['text']
                noised = add_noise(text, rng, args)
                outdata = { 'input': noised, 'output': text }
                print(json.dumps(outdata, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
