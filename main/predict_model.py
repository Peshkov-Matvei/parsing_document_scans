import torch
from datasets import load_from_disk
from transformers import AutoProcessor, AutoModelForTokenClassification
from PIL import ImageDraw, ImageFont

input_dir = 'data/funsd_layoutlmv3'
model_dir = 'model/checkpoint-1000'

test = load_from_disk(f'{input_dir}/test')

processor = AutoProcessor.from_pretrained('microsoft/layoutlmv3-base',
                                          apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

example = test[1]

image = example['image']
words = example['tokens']
boxes = example['bboxes']
word_labels = example['ner_tags']

encoding = processor(image, words, boxes=boxes,
                     word_labels=word_labels, return_tensors='pt')

with torch.no_grad():
    outputs = model(**encoding)

logits = outputs.logits

predictions = logits.argmax(-1).squeeze().tolist()
labels = encoding.labels.squeeze().tolist()


def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


token_boxes = encoding.bbox.squeeze().tolist()
width, height = image.size

true_predictions = [model.config.id2label[pred]
                    for pred, label in zip(predictions, labels)
                    if label != -100]
true_labels = [model.config.id2label[label]
               for pred, label in zip(predictions, labels) if label != -100]
true_boxes = [unnormalize_box(box, width, height)
              for box, label in zip(token_boxes, labels) if label != -100]

draw = ImageDraw.Draw(image)
font = ImageFont.load_default()


def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label


label2color = {
    'question': 'blue',
    'answer': 'green',
    'header': 'orange',
    'other': 'violet'
    }

for prediction, box in zip(true_predictions, true_boxes):
    predicted_label = iob_to_label(prediction).lower()
    draw.rectangle(box, outline=label2color[predicted_label])
    draw.text((box[0] + 10, box[1] - 10), text=predicted_label,
              fill=label2color[predicted_label], font=font)

image = example['image']
image = image.convert('RGB')

draw = ImageDraw.Draw(image)

for word, box, label in zip(example['tokens'],
                            example['bboxes'], example['ner_tags']):
    actual_label = iob_to_label(model.config.id2label[label]).lower()
    box = unnormalize_box(box, width, height)
    draw.rectangle(box, outline=label2color[actual_label], width=2)
    draw.text((box[0] + 10, box[1] - 10), actual_label,
              fill=label2color[actual_label], font=font)

image.show()
