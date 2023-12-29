# Fine-tuning Resnet 50 Model

## Setup

Install the required packages from the requirements file into your virtual environment
```shell
>>> source .venv/bin/activate
>>> python -m pip install -r requirements.txt
```

## Usage

to train a model simply run:
```shell
>>> python train.py --images_url <url to images zip> --annotations_url <url of annotations file in coco format>
```

to infer trained model:
```shell
>>> python test.py --image_path <path to test image>
```
