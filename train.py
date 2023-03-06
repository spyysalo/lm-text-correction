#!/usr/bin/python3

import sys
import json
import random

from argparse import ArgumentParser

from sentence_splitter import SentenceSplitter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    pipeline,
)


TEMPLATE = '''Korjaa teksti.

Teksti: {}

Korjattu: {}

Valmis.
'''


def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('data')
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--seed', type=int, default=42)
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

    if args.tokenizer is None:
        args.tokenizer = args.model
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    docs = load_documents(args.data)
    train_docs, valid_docs = split_documents(docs, seed=args.seed)

    train_sents = split_into_sentences(
        train_docs,
        max_chars=250,
        #max_lines=100000,
        #max_sents=100000,
        seed=args.seed
    )
    valid_sents = split_into_sentences(
        valid_docs,
        max_chars=250,
        max_lines=1000,
        max_sents=1000,
        seed=args.seed
    )

    train_examples = make_examples(train_sents)
    valid_examples = make_examples(valid_sents)

    train_data = Dataset.from_dict({ 'text': train_examples })
    valid_data = Dataset.from_dict({ 'text': valid_examples })

    tokenize = lambda d: tokenizer(d['text'], truncation=True)
    train_data = train_data.map(tokenize)
    valid_data = valid_data.map(tokenize)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='output',
        evaluation_strategy='steps',
        logging_strategy='steps',
        learning_rate=1e-5,    # 2e-5
        weight_decay=0.01,
        eval_steps=1000,
        logging_steps=1000,
        save_steps=5000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        max_steps=100000,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
    )

    trainer.train()

    valid_results = trainer.evaluate(valid_data)
    print('FINAL VALIDATION LOSS:', valid_results['eval_loss'])

    trainer.save_model('trained-model')
    
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    text = 'Korjaa teksti.\n\nTeksti: Turu sntyi urajoen sulle j ene 1200lukua a s o Smen anhinkaunki.\n\nKorjattu:'

    print(pipe(text, max_new_tokens=25)[0]['generated_text'])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
