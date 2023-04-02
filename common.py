import numpy as np

import datasets
import evaluate


def load_data(train_fn, valid_fn, max_train=None, max_valid=None):
    train_dataset = datasets.Dataset.from_json(train_fn)
    valid_dataset = datasets.Dataset.from_json(valid_fn)

    # Note! Intentionally not shuffling
    if max_train is not None and len(train_dataset) > max_train:
        train_dataset = train_dataset.select(range(max_train))
    if max_valid is not None and len(valid_dataset) > max_valid:
        valid_dataset = valid_dataset.select(range(max_valid))

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset,
    })

    return dataset


def compute_metrics_for_texts(predictions, references):
    char_metric = evaluate.load('character')
    word_metric = evaluate.load('wer')

    for i in range(10):
        print('GOLD:', references[i])
        print('PRED:', predictions[i])
        print('---')

    try:
        char_result = char_metric.compute(
            predictions=predictions,
            references=references
        )
    except:
        char_result = { 'cer_score': float('inf') }

    try:
        word_result = word_metric.compute(
            predictions=predictions,
            references=references
        )
    except:
        word_result = { 'wer': float('inf') }

    return {
        'cer_score': char_result['cer_score'],
        'wer': word_result,
    }
