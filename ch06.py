#通过微调遵循人类指令6.5
#先为有监督微调准备数据集
import json
import os
import urllib
from importlib.metadata import version

pkgs = [
    "numpy",
    "matplotlib",
    "tiktoken",
    "torch",
    "tqdm",
    "tensorflow",
]
for p in pkgs:
    print(f"{p} version:{version(p)}")
    
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.utlopen(url) as response:
            text_data= response.read().decode("utf-8")
        with open(file_path, "w", encoding = "stf-8") as file:
            file.write(text_data)
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
file_path = "instruction-data.json"
url = (
    ""
)
