import os
import sys
sys.path.append(".")
from synthesize import synthesize, get_model, get_vocoder, preprocess_mandarin, preprocess_vietnamese, preprocess_english
import yaml
import argparse
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
args = parser.parse_args()
# set args
args.lexicon = "/src/lexicon/vlsp_2021-lexicon.txt"
args.restore_step = 300000
args.mode = "single"
args.speaker_id = 0
args.preprocess_config = "/src/config/vlsp_2021/preprocess.yaml"
args.model_config = "/src/config/vlsp_2021/model.yaml"
args.train_config = "/src/config/vlsp_2021/train.yaml"
args.pitch_control = 1.0
args.energy_control = 1.0
args.duration_control = 1.0

# config
preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

# Get model
model = get_model(args, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device)

control_values = args.pitch_control, args.energy_control, args.duration_control


with open(args.lexicon, "r", encoding="utf-8") as f:
    texts = [line.strip("\n").split("\t")[0] + '.' for line in f.readlines()]


if __name__ == "__main__":

    for text in texts:
        ids = raw_texts = [text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            pre_text = np.array([preprocess_english(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            pre_text = np.array([preprocess_mandarin(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "vi":
            pre_text = np.array([preprocess_vietnamese(text, preprocess_config)])
        text_lens = np.array([len(pre_text[0])])
        batchs = [(ids, raw_texts, speakers, pre_text, text_lens, max(text_lens))]

        synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)