from transformers import ResNetForImageClassification, AutoImageProcessor
from PIL import Image

import argparse
import torch


def run_inference(model_path: str, image: Image.Image):
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained(model_path)

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # model predicts one of the 1000 ImageNet classes
    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process user input for a testing model.")

    # Add arguments
    parser.add_argument("--model_path", default="./results", help="Specify the path to the model.")
    parser.add_argument("--image_path", default=None, help="Specify the path to the image for inference.")

    # Parse the arguments
    args = parser.parse_args()

    # load pillow image
    image = Image.open(args.image_path)

    # get inference results
    print("result:", run_inference(args.model_path, image))
