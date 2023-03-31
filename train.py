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
T_START = 'Input:\n'

T_MIDDLE = '\n\nOutput:\n'

#MAX_LEN = 512 # TODO


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--learning-rate', type=float, default=5e-05)
    ap.add_argument('--max-train-examples', type=int, default=None)
    ap.add_argument('--max-valid-examples', type=int, default=None)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('model')
    ap.add_argument('train_data')
    ap.add_argument('valid_data')
    return ap


def preprocess(data, tokenizer):
    input_texts = data['input']
    output_texts = data['output']

    templated = []
    for i, o in zip(input_texts, output_texts):
        prompt = T_START + i + T_MIDDLE + tokenizer.bos_token
        output = o + tokenizer.eos_token
        templated.append(prompt+output)

    # Truncation would be problematic for this task
    tokenized = tokenizer(templated, truncation=False)

    return tokenized


def get_outputs(ref_ids, pred_ids, tokenizer):
    ref_ids, pred_ids = ref_ids.tolist(), pred_ids.tolist()

    # remove prompts (everything up to the first bos in labels)
    for i in range(len(ref_ids)):
        o = ref_ids[i].index(tokenizer.bos_token_id)
        ref_ids[i] = ref_ids[i][o+1:]
        pred_ids[i] = pred_ids[i][o:]    # labels are shifted + 1

    # remove everything starting at the first eos
    for i in range(len(ref_ids)):
        try:
            o = ref_ids[i].index(tokenizer.eos_token_id)
            ref_ids[i] = ref_ids[i][:o]
        except:
            warning(f'missing eos in refs {i}')

    for i in range(len(pred_ids)):
        try:
            o = pred_ids[i].index(tokenizer.eos_token_id)
            pred_ids[i] = pred_ids[i][:o]
        except:
            pass    # preds don't necessarily have eos


    return ref_ids, pred_ids


def compute_metrics(preds_and_refs, tokenizer):
    pred_ids, ref_ids = preds_and_refs

    # -100 can't be decoded, so replace with pad id
    ref_ids = np.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)

    ref_ids, pred_ids = get_outputs(ref_ids, pred_ids, tokenizer)

    preds = tokenizer.batch_decode(pred_ids) #, skip_special_tokens=True)
    refs = tokenizer.batch_decode(ref_ids) #, skip_special_tokens=True)

    return compute_metrics_for_texts(preds, refs)


def logits_argmax(logits, labels):
    # https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(axis=-1)


class PromptMaskingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)

        bos_token_id = self.tokenizer.bos_token_id
        for i in range(len(data['labels'])):
            bos_indices = np.where(data['labels'][i] == bos_token_id)[0]
            if len(bos_indices) > 0:
                # TODO this should really be bos_indices[0]+1 but that
                # would mask the BOS which would mess up the current
                # logic for separating the prompt from the output
                data['labels'][i,:bos_indices[0]] = -100
            else:
                warning('missing BOS/-100 in labels')

        return data


def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    #tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(args.model)
    #model.max_length = MAX_LEN

    # If we don't have a pad_token, add a new one. Note that we want
    # to avoid reusing eos_token or bos_token because that would lead
    # to these being replaced with -100 in labels, which would mess
    # with the logic for isolating the output from the prompt.
    # (ditto for unk_token b/c that can match bos or eos.)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))

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

    if args.max_train_examples is not None:
        limit = args.max_train_examples
        dataset['train'] = dataset['train'].select(range(limit))

    dataset = dataset.map(
        lambda d: preprocess(d, tokenizer),
        batched=True
    )

    for s in ('train', 'validation'):
        print(f'max {s} input_ids length',
              max(len(i) for i in dataset[s]['input_ids']))

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        output_dir='output',
        logging_dir='logs',
        per_device_train_batch_size=8,
        #gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        #eval_accumulation_steps=1,
        evaluation_strategy='steps',
        logging_strategy='steps',
        weight_decay=0.01,
        num_train_epochs=1,
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
