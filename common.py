import numpy as np

import datasets
import evaluate


def load_data(fn, seed=42):
    dataset = datasets.load_dataset('json', data_files=fn)

    split = dataset['train'].train_test_split(
        test_size=1000,
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


def compute_metrics(preds_and_refs, tokenizer):
    pred_ids, ref_ids = preds_and_refs

    # -100 can't be decoded, so replace with pad id
    ref_ids = np.where(ref_ids != -100, ref_ids, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(ref_ids, skip_special_tokens=True)

    return compute_metrics_for_texts(preds, refs)
