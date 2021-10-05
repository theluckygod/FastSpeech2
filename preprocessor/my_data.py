import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import logging

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    # prepare audio
    speaker_dirs = []
    for set in os.listdir(in_dir):
        set_path = os.path.join(in_dir, set)
        if not os.path.isdir(set_path):
            continue
        for speaker in os.listdir(os.path.join(set_path, "waves")):
            speaker_dirs.append(os.path.join(set_path, "waves", speaker))

    for speaker_dir in tqdm(speaker_dirs):
        speaker = os.path.basename(speaker_dir)
        os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
        for file_name in os.listdir(speaker_dir):
            wav_path = os.path.join(speaker_dir, file_name)
            try:
                wav, _ = librosa.load(wav_path, sampling_rate)
            except:
                logging.warn(f"Audio {wav_path} can not read!")
                continue
            wav = wav / max(abs(wav)) * max_wav_value
            try:
                wavfile.write(
                    os.path.join(out_dir, speaker, file_name),
                    sampling_rate,
                    wav.astype(np.int16),
                )
            except:
                logging.warn(f"Audio {os.path.join(out_dir, speaker, file_name)} can not write!")
                continue

    # prepare text
    for set in os.listdir(in_dir):
        set_path = os.path.join(in_dir, set)
        if not os.path.isdir(set_path):
            continue

        text_path = os.path.join(set_path, "prompts.txt")
        with open(text_path, encoding="utf8") as f:
            for text in f:
                text = text.strip("\n")
                file_name = text.split(" ")[0]
                label = " ".join(text.split(" ")[1:])
                label = _clean_text(label, cleaners)

                speaker = file_name.split("_")[0]
                if os.path.isfile(os.path.join(out_dir, speaker, f"{file_name}.wav")):
                    with open(os.path.join(out_dir, speaker, f"{file_name}.lab"), "w", encoding='utf-8') as f1:
                        f1.write(label)
                else:
                    logging.warn(f"Audio {os.path.join(out_dir, speaker, f'{file_name}.wav')} was not found!")