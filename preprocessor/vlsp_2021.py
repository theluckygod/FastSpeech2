import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import logging
from joblib import Parallel, delayed

from text import _clean_text

def process_wav(out_dir, in_dir, utt, speaker, sampling_rate, max_wav_value):
    wav_path = os.path.join(in_dir, f"{utt}.wav")
    wav, _ = librosa.load(wav_path, sr=sampling_rate)
    wav = wav / max(abs(wav)) * max_wav_value
    wavfile.write(
        os.path.join(out_dir, speaker, f"{speaker}-{utt}.wav"),
        sampling_rate,
        wav.astype(np.int16),
    )

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    utterances = [file[:-4] for file in os.listdir(in_dir) if file[-4:] == ".wav"]

    speaker = "female" # only female speaker
    os.makedirs(os.path.join(out_dir, speaker) , exist_ok=True)

    jobs = []
    for utt in tqdm(utterances):
        # preprocess audio
        jobs.append(delayed(process_wav)(out_dir, in_dir, utt, speaker, sampling_rate, max_wav_value))
    Parallel(n_jobs=6, verbose=1)(jobs)

    for utt in tqdm(utterances):
        # preprocess text
        txt_path = os.path.join(in_dir, f"{utt}.txt")
        with open(txt_path) as f:
            text = f.readline().strip("\n")
            text = _clean_text(text, cleaners)

        with open(
            os.path.join(out_dir, speaker, f"{speaker}-{utt}.lab"),
            "w",
        ) as f1:
            f1.write(text)
