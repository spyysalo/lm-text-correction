import numpy as np

import datasets
import evaluate


def load_data(fn, test_size=1000, seed=42):
    dataset = datasets.load_dataset('json', data_files=fn)

    split = dataset['train'].train_test_split(
        test_size=test_size,
        shuffle=False,    # in case data is ordered split sentences
        seed=seed,
    )

    # rename
    dataset = datasets.DatasetDict({
        'train': split['train'],
        'validation': split['test'],
    })

    dataset = dataset.shuffle(seed=seed)

    return dataset


def compute_metrics_for_texts(predictions, references):
    char_metric = evaluate.load('character')
    word_metric = evaluate.load('wer')

    print(predictions[:10])
    print(references[:10])

    char_result = char_metric.compute(
        predictions=predictions,
        references=references
    )

    word_result = word_metric.compute(
        predictions=predictions,
        references=references
    )

    return {
        'cer_score': char_result['cer_score'],
        'wer': word_result,
    }
