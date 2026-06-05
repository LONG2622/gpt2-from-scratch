# gpt_download.py
import os
import json
import requests
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size, models_dir):
    allowed_sizes = ["124M", "355M", "774M", "1558M"]
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    ]

    os.makedirs(model_dir, exist_ok=True)

    for filename in filenames:
        file_path = os.path.join(model_dir, filename)
        url = f"{base_url}/{model_size}/{filename}"

        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            with open(file_path, "wb") as f:
                for chunk in tqdm(response.iter_content(chunk_size=1024)):
                    f.write(chunk)

    with open(os.path.join(model_dir, "hparams.json")) as f:
        hparams = json.load(f)
    params = {}
    try:
        ckpt = tf.train.latest_checkpoint(model_dir)
        init_vars = tf.train.list_variables(ckpt)
        for name, shape in init_vars:
            array = tf.train.load_variable(ckpt, name)
            params[name] = array
    except Exception as e:
        print(f"TensorFlow loading failed: {e}")
        print("Please make sure you have tensorflow installed: pip install tensorflow")

    return hparams, params