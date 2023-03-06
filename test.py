#!/usr/bin/python3

import sys

from argparse import ArgumentParser

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    ap.add_argument('model')
    return ap


def load_documents(fn):
    documents = []
    with open(fn) as f:
        for l in f:
            documents.append(json.loads(l))
    print(f'loaded {len(documents)} documents from {fn}', file=sys.stderr)
    return documents


def split_documents(documents, train_ratio=0.95, seed=None):
    documents = documents[:]
    random.seed(seed)
    random.shuffle(documents)
    idx = int(len(documents)*train_ratio)
    return documents[:idx], documents[idx:]


def split_into_sentences(documents, max_chars=None, max_lines=None,
                         max_sents=None, seed=None):
    lines = [l for d in documents for l in d['text'].split('\n')]
    lines = [l for l in lines if l and not l.isspace()]
    
    random.seed(seed)
    random.shuffle(lines)

    if max_lines is not None:
        lines = lines[:max_lines]

    splitter = SentenceSplitter(language='fi')

    sentences = [s for l in lines for s in splitter.split(l)]
    sentences = [s for s in sentences if s and not s.isspace()]

    if max_chars is not None:
        sentences = [s for s in sentences if len(s) <= max_chars]
    
    random.shuffle(sentences)
    if max_sents is not None:
        sentences = sentences[:max_sents]
    
    return sentences


def add_errors(text, p=0.1):
    chars = []
    for c in text:
        if random.random() > p:
            chars.append(c)
    text = ''.join(chars)
    return text


def make_examples(texts, p=0.1):
    examples = []
    for t in texts:
        e = add_errors(t, p)
        examples.append(TEMPLATE.format(e, t))
    return examples


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    text = 'Korjaa teksti.\n\nTeksti: Turu sntyi urajoen sulle j enen 1200lukua ja s o Sumen anhinkaunki.\n\nKorjattu:'

    print(pipe(text, max_new_tokens=25)[0]['generated_text'])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
