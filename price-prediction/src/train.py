from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from transformers import (
    DistilBertTokenizerFast, TFDistilBertForSequenceClassification, TFTrainer,
    TFTrainingArguments
)


from prep_data import load_data

tokenizer_config = {
    'truncation': True,
    'padding': 'max_length',
    'max_length': 128
}
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')


def prepare_dataset(dataset, n_examples):
    text, prices = load_data(dataset, n_examples)
    encoded_text = tokenizer(text, **tokenizer_config)
    return tf.data.Dataset.from_tensor_slices((
        dict(encoded_text),
        prices
    ))


def train():
    train_dataset = prepare_dataset('train', 100000)
    val_dataset = prepare_dataset('validate', 30000)

    training_args = TFTrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-cased", num_labels=1
        )

    trainer = TFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    trainer.save_model()

    return trainer


def evaluate(trainer):
    test_dataset = prepare_dataset('test', 30000)
    preds = trainer.predict(test_dataset)
    labels = [1, 2, 3, 4]

    conf_matrix = confusion_matrix(
        preds.label_ids, preds.predictions.round(), labels=labels
    )
    (
        ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
        .plot().figure_.savefig('conf-matrix.png')
    )


if __name__ == '__main__':
    trainer = train()
    evaluate(trainer)
