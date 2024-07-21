import numpy as np
from datasets import load_from_disk
from datasets.features import ClassLabel
from datasets import Features, Sequence, Value, Array2D, Array3D
from transformers import (LayoutLMv3ForTokenClassification,
                          TrainingArguments, Trainer)
from transformers import AutoProcessor
from transformers.data.data_collator import default_data_collator
from evaluate import load

input_dir = 'data/funsd_layoutlmv3'

train = load_from_disk(f'{input_dir}/train')
test = load_from_disk(f'{input_dir}/test')

processor = AutoProcessor.from_pretrained('microsoft/layoutlmv3-base',
                                          apply_ocr=False)

features = train.features
column_names = train.column_names
image_column_name = 'image'
text_column_name = 'tokens'
boxes_column_name = 'bboxes'
label_column_name = 'ner_tags'


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
else:
    label_list = get_label_list(train[label_column_name])
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}

num_labels = len(label_list)


def prepare_examples(examples):
    images = examples[image_column_name]
    words = examples[text_column_name]
    boxes = examples[boxes_column_name]
    word_labels = examples[label_column_name]

    encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
                         truncation=True, padding='max_length')
    return encoding


features = Features({
    'pixel_values': Array3D(dtype='float32', shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype='int64', shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int64'))
})

train_dataset = train.map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)

eval_dataset = test.map(
    prepare_examples,
    batched=True,
    remove_columns=column_names,
    features=features,
)

train_dataset.set_format('torch')

metric = load('seqeval')

return_entity_level_metrics = False


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions,
                             references=true_labels)
    if return_entity_level_metrics:
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f'{key}_{n}'] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            'precision': results['overall_precision'],
            'recall': results['overall_recall'],
            'f1': results['overall_f1'],
            'accuracy': results['overall_accuracy']
        }


model = LayoutLMv3ForTokenClassification.from_pretrained(
    'microsoft/layoutlmv3-base',
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir='model',
    max_steps=1000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    evaluation_strategy='steps',
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
