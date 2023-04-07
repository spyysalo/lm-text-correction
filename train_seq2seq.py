#!/usr/bin/env python3

# Train sequence-to-sequence model for error correction

import sys
import json
import os

import numpy as np

from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from common import load_data, compute_metrics_for_texts


# Avoid "huggingface/tokenizers: The current process just got forked" warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prefix to add to all inputs
PREFIX = 'Korjaa virheet: '

MAX_LEN = 512 # TODO


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--learning-rate', type=float, default=5e-05)
    ap.add_argument('--max-train-examples', type=int, default=None)
    ap.add_argument('--max-valid-examples', type=int, default=None)
    ap.add_argument('model')
    ap.add_argument('train_data')
    ap.add_argument('valid_data')
    return ap


def preprocess(data, tokenizer):
    input_texts = data['input']
    output_texts = data['output']

    input_texts = [PREFIX + i for i in input_texts]

    tokenizer_args = {
        'truncation': True,
        'max_length': MAX_LEN,
    }

    inputs = tokenizer(input_texts, **tokenizer_args)
    outputs = tokenizer(text_target=output_texts, **tokenizer_args)

    inputs['labels'] = outputs['input_ids']
    return inputs


def compute_metrics(preds_and_refs, tokenizer):
    pred_ids, ref_ids = preds_and_refs

    # -100 can't be decoded, so replace with pad id
    ref_ids = np.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

    return compute_metrics_for_texts(preds, refs)


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    dataset = load_data(
        args.train_data,
        args.valid_data,
        max_train=args.max_train_examples,
        max_valid=args.max_valid_examples,
    )

    # report validation metrics for just copying input
    result = compute_metrics_for_texts(
        predictions=dataset['validation']['input'],
        references=dataset['validation']['output'],
    )
    print('validation metrics for copy:', result)

    dataset = dataset.map(
        lambda d: preprocess(d, tokenizer),
        batched=True
    )

    for s in ('train', 'validation'):
        print(f'max {s} input_ids length',
              max(len(i) for i in dataset[s]['input_ids']))

    train_args = Seq2SeqTrainingArguments(
        learning_rate=args.learning_rate,
        output_dir='output',
        logging_dir='logs',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        evaluation_strategy='steps',
        logging_strategy='steps',
        num_train_epochs=1,
        logging_steps=1000,
        eval_steps=1000,
        save_strategy='no',
        #save_total_limit=5,
        #save_steps=1000,
    )
    #print(train_args)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=lambda o: compute_metrics(o, tokenizer),
    )

    trainer.train()

    valid_results = trainer.evaluate(dataset['validation'])
    print('FINAL VALIDATION LOSS:', valid_results['eval_loss'])
    print('FINAL VALIDATION CER:', valid_results['eval_cer_score'])
    print('FINAL VALIDATION WER:', valid_results['eval_wer'])

    trainer.save_model('trained-seq2seq-model')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
