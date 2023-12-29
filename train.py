from transformers import ResNetForImageClassification, Trainer, TrainingArguments

import argparse
import os

from prepare_dataset import prepare_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process user input for a training task.")

    # Add arguments
    parser.add_argument("--tokenizer", default="microsoft/resnet-50", help="Specify the tokenizer.")
    parser.add_argument("--epochs", type=int, default=10, help="Specify the number of epochs.")
    parser.add_argument("--output_path", default="./results", help="Specify the output path.")
    parser.add_argument("--images_url", default=None, help="Specify the images URL.")
    parser.add_argument("--annotations_url", default=None, help="Specify the annotations URL.")

    # Parse the arguments
    args = parser.parse_args()

    return args


def initiate_ic_training(training_dataset, eval_dataset, tokenizer: str,
                         epochs: int, output_path: str):
    # setup training arguments
    training_args = TrainingArguments(
        output_dir=output_path,  # output directory
        num_train_epochs=epochs,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=32,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        logging_steps=10,
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True
    )

    print("--- dataset setup successfully ---\n--- setting up model ---")
    model = ResNetForImageClassification.from_pretrained(tokenizer)

    # setup model trainer
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=training_dataset,  # training dataset
        eval_dataset=eval_dataset  # evaluation dataset
    )

    print("--- starting model training ---\n")
    trainer.train()

    print(f"--- model training successful ---\n--- model saved to ./{output_path} ---")
    trainer.save_model(output_path)


def main():
    args = parse_arguments()
    os.makedirs(args.output_path, exist_ok=True)

    # generate dataset
    train_dataset, val_dataset, categories = prepare_dataset(
        tokenizer=args.tokenizer, train_size=0.8, test_size=0.2,
        images_url=args.images_url, annotations_url=args.annotations_url
    )

    # begin training
    initiate_ic_training(
        train_dataset, val_dataset, tokenizer=args.tokenizer,
        epochs=args.epochs, output_path=args.output_path
    )


if __name__ == "__main__":
    main()
