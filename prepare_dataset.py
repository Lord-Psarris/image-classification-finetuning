from transformers import AutoImageProcessor, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from PIL import Image

import zipfile
import torch
import json
import wget
import os


def prepare_dataset(tokenizer: str, images_url: str, annotations_url: str, train_size: float, test_size: float):
    """
    generate training & validation dataset splits from dataset for image classification

    :param tokenizer:
    :param images_url:
    :param annotations_url:
    :param train_size:
    :param test_size:

    :return tuple: training and validation datasets
    """
    # parse dataset
    train_images, val_images, train_labels, val_labels, categories = download_dataset(
        images_url=images_url, annotations_url=annotations_url, test_size=test_size, train_size=train_size
    )

    # set processor
    image_processor = AutoImageProcessor.from_pretrained(tokenizer)

    # generate encodings
    print("\n--- encoding dataset ---")
    train_encodings = image_processor(train_images, truncation=True, padding=True)
    val_encodings = image_processor(val_images, truncation=True, padding=True)

    # generate dataset
    print("\n--- preparing encoded dataset ---")
    train_dataset = BaseDataset(train_encodings, train_labels)
    val_dataset = BaseDataset(val_encodings, val_labels)

    return train_dataset, val_dataset, categories


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def download_dataset(images_url: str, annotations_url: str, train_size: float, test_size: float):
    print("\n---- downloading dataset ----")
    # download_images
    wget.download(images_url, "images.zip")
    images_path = parse_dataset_images("images.zip")

    # download annotations
    wget.download(annotations_url, "annotations.json")
    base_labels, categories = parse_dataset_annotations("annotations.json")

    # process dataset
    print("\n---- processing dataset ----")

    images = []
    labels = []

    for i in base_labels:
        # append image
        path = os.path.join(images_path, i["image"]["file_name"])
        images.append(Image.open(path).convert("RGB"))

        # append label
        labels.append(i["category"])

    # split dataset
    print("\n---- splitting dataset ----")
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                          train_size=train_size,
                                                                          test_size=test_size)
    return train_images, val_images, train_labels, val_labels, categories


def parse_dataset_images(zip_path: str) -> str:
    # configure model path
    images_path = ".datasets/images"
    os.makedirs(images_path, exist_ok=True)

    # unzip content
    unzip_folder(zip_path, output_folder=images_path)

    # delete cleared files
    os.remove(zip_path)
    return images_path


def parse_dataset_annotations(json_path):
    with open(json_path) as file:
        data = json.load(file)

    # get data content
    images = data["images"]
    categories = data["categories"]
    annotations = data["annotations"]

    labels = []
    for i in annotations:
        image = next(iter(filter(lambda x: x["id"] == i["image_id"], images)), None)
        if image is None:
            continue

        labels.append({
            "image": image,
            "category": i["category_id"],
        })

    # remove annotations file
    os.remove(json_path)

    # return labels (images_path, category id), categories
    return labels, categories


def unzip_folder(zip_file_path: str, output_folder: str = ""):
    # extract file
    try:
        print("\n---- unzipping file ----")
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
            print(f"Successfully unzipped {zip_file_path} to the ./{output_folder} folder")

    except Exception as e:
        print(f"Error unzipping {zip_file_path}: {str(e)}")
