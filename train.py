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
        prompt = T_START + i + T_MIDDLE + tokenizer.bos_token
        output = o + tokenizer.eos_token
        templated.append(prompt+output)

    tokenized = tokenizer(
        templated,
        truncation=True,
        max_length=MAX_LEN
    )

    return tokenized


def trim_to_output(text, tokenizer):
    orig_text = text
    if tokenizer.bos_token in text:
        text = text.split(tokenizer.bos_token)[-1]
    if tokenizer.eos_token in text:
        text = text.split(tokenizer.eos_token)[0]
    if not text:
        warning(f'emptied: {orig_text}')
    return text


def compute_metrics(preds_and_refs, tokenizer):
    pred_ids, ref_ids = preds_and_refs

    # -100 can't be decoded, so replace with pad id
    ref_ids = np.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(pred_ids) #, skip_special_tokens=True)
    refs = tokenizer.batch_decode(ref_ids) #, skip_special_tokens=True)

    refs = [trim_to_output(t, tokenizer) for t in refs]
    preds = [trim_to_output(t, tokenizer) for t in preds]

    return compute_metrics_for_texts(preds, refs)


def logits_argmax(logits, labels):
    # https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(axis=-1)


class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)

        # -100 labels for prompt (everything up to and including bos)
        bos_token = self.tokenizer.bos_token
        bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)

        # https://github.com/pytorch/pytorch/issues/9413#issuecomment-406030626
        is_bos_token_id = (data['labels'] == bos_token_id)
        bos_indices = is_bos_token_id.nonzero()
        for i, j in bos_indices.tolist():
            # TODO this should really be j+1 but that would mask the
            # BOS which would mess up the current implementation of
            # trim_to_output
            data['labels'][i,:j] = -100

        return data


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    #tokenizer.padding_side = 'right'

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
        #eval_accumulation_steps=1,
        evaluation_strategy='steps',
        logging_strategy='steps',
        weight_decay=0.01,
        eval_steps=1000,
        logging_steps=1000,
        save_strategy='no',
        #save_total_limit=5,
        #save_steps=1000,
    )

    data_collator = PromptMaskingDataCollator(
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
        preprocess_logits_for_metrics=logits_argmax,
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

    text = T_START + 'Turu sntyi urajoen sulle j ene 1200lukua a s o Smen anhinkaunki.' + T_MIDDLE + tokenizer.bos_token

    print(pipe(text, max_new_tokens=25)[0]['generated_text'])


if __name__ == '__main__':
    sys.exit(main(sys.argv))
