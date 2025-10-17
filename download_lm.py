import gzip
import os

import wget
from huggingface_hub import hf_hub_download


def fetch_language_model():
    home_dir = os.path.expanduser("~")
    lm_dir = os.path.join(home_dir, "language_models")
    final_lm_file = os.path.join(lm_dir, "4gram_lowercase.arpa")

    if os.path.isfile(final_lm_file):
        return

    print("Starting download")

    model_url = "http://www.openslr.org/resources/11/4-gram.arpa.gz"
    original_dir = os.getcwd()
    os.chdir(lm_dir)

    downloaded_archive = wget.download(model_url)

    temp_model = "model_uppercase.arpa"
    with gzip.open(downloaded_archive, "rb") as gz_file:
        content = gz_file.read()
        with open(temp_model, "wb") as out_file:
            out_file.write(content)

    with open(temp_model, "r", encoding="utf-8") as source:
        with open("4gram_lowercase.arpa", "w", encoding="utf-8") as target:
            for text_line in source:
                target.write(text_line.lower())

    os.chdir(original_dir)

    print(f"LM saved at {final_lm_file}")


def fetch_conformer_model_weights():
    checkpoint_path = hf_hub_download(
        repo_id="NickolayFM/conformer-ASR",
        filename="checkpoint-epoch84.pth",
        local_dir="./conformer_30m",
    )

    print(f"Checkpoint saved at: {checkpoint_path}")
