import os
import sys
sys.path.append(".")
from synthesize import synthesize, get_model, get_vocoder, preprocess_mandarin, preprocess_vietnamese, preprocess_english
import yaml
import argparse
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
args = parser.parse_args()
# set args
args.testcases = "test/vi_testcases.txt"
args.lexicon = "lexicon/vlsp_2021-lexicon.txt"
args.restore_step = 100000
args.mode = "batch"
args.speaker_id = 27
args.preprocess_config = "config/vlsp_2021/preprocess.yaml"
args.model_config = "config/vlsp_2021/model.yaml"
args.train_config = "config/vlsp_2021/train.yaml"
args.pitch_control = 1.0
args.energy_control = 1.0
args.duration_control = 1.0

# config
preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
preprocess_config["path"]["preprocessed_path"] = "../../data/align/exp_vivos_fastspeech2_v0_paddle_mfa1"
model_config["path"]["vocoder_path"]["custom"] = "../../checkpoints/hifigan/generator_vivos.pth.tar"
model_config["vocoder"]["model"] = "HiFi-GAN"
model_config["vocoder"]["speaker"] = "custom"
train_config["path"]["ckpt_path"] = "../../checkpoints/exp_vivos_fastspeech2_v0_paddle_mfa1/ckpt"
train_config["path"]["log_path"] = "../../checkpoints/exp_vivos_fastspeech2_v0_paddle_mfa1/log"
train_config["path"]["result_path"] = "../../checkpoints/exp_vivos_fastspeech2_v0_paddle_mfa1/result"

configs = (preprocess_config, model_config, train_config)

# Get model
model = get_model(args, configs, device, train=False)

# Load vocoder
vocoder = get_vocoder(model_config, device, sr=preprocess_config["preprocessing"]["audio"]["sampling_rate"])

control_values = args.pitch_control, args.energy_control, args.duration_control


with open(args.testcases, "r", encoding="utf-8") as f:
    texts = [line.strip("\n") for line in f.readlines()]


if __name__ == "__main__":
    batch_size = train_config["optimizer"]["batch_size"]
    batchs = []
    for idx, text in enumerate(tqdm(texts)):
        ids = raw_texts = [text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            pre_text = np.array([preprocess_english(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            pre_text = np.array([preprocess_mandarin(text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "vi":
            pre_text = np.array([preprocess_vietnamese(text, preprocess_config)])
        text_lens = np.array([len(pre_text[0])])

        batchs.append((ids, raw_texts, speakers, pre_text, text_lens, max(text_lens)))
        if len(batchs) == batch_size or idx == len(texts) - 1:
            synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
            batchs = []