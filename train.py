#!/usr/bin/python3

# Training causal model for error correction

import sys
import os

import numpy as np

from logging import warning
from argparse import ArgumentParser

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

from common import load_data, compute_metrics_for_texts


# Avoid "huggingface/tokenizers: The current process just got forked" warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Template for formatting data
T_START = '''Korjaa virheet:

Teksti:
'''

T_MIDDLE = '''

Korjattu:
'''

T_END = '''

Valmis.'''

MAX_LEN = 512 # TODO


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--learning-rate', type=float, default=5e-05)
    ap.add_argument('--max-train-examples', type=int, default=None)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('model')
    ap.add_argument('data')
    return ap


def preprocess(data, tokenizer):
    input_texts = data['input']
    output_texts = data['output']

    templated = []
    for i, o in zip(input_texts, output_texts):
        templated.append(T_START + i + T_MIDDLE + o + T_END)
        
    tokenized = tokenizer(
        templated,
        truncation=True,
        max_length=MAX_LEN
    )

    return tokenized


def trim_to_output(text):
    if text.startswith(T_START):
        text = text[len(T_START):]
    if T_MIDDLE in text:
        text = text.split(T_MIDDLE)[1]
    if T_END in text:
        text = text.split(T_END)[0]
    return text

        
def compute_metrics(preds_and_refs, tokenizer):
    preds, ref_ids = preds_and_refs
    pred_ids = preds.argmax(axis=-1)

    # -100 can't be decoded, so replace with pad id
    ref_ids = np.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

    refs = [trim_to_output(t) for t in refs]
    preds = [trim_to_output(t) for t in preds]
    
    return compute_metrics_for_texts(preds, refs)


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.max_length = MAX_LEN
    
    dataset = load_data(args.data)

    # report validation metrics for just copying input
    result = compute_metrics_for_texts(
        predictions=dataset['validation']['input'],
        references=dataset['validation']['output'],
    )
    print('validation metrics:', result)

    if args.max_train_examples is not None:
        limit = args.max_train_examples
        dataset['train'] = dataset['train'].select(range(limit))

    dataset = dataset.map(
        lambda d: preprocess(d, tokenizer),
        batched=True
    )

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        output_dir='output',
        logging_dir='logs',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        eval_accumulation_steps=1,
        evaluation_strategy='steps',
        logging_strategy='steps',
        weight_decay=0.01,
        eval_steps=1000,
        logging_steps=1000,
        save_strategy='no',
        #save_total_limit=5,
        #save_steps=1000,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
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

    trainer.save_model('trained-model')

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=model.device
    )

    text = T_START + 'Turu sntyi urajoen sulle j ene 1200lukua a s o Smen anhinkaunki.' + T_MIDDLE

    print(pipe(text, max_new_tokens=25)[0]['generated_text'])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
