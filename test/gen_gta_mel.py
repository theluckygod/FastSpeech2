import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
import logging

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gen_GTA_mel(model, dataloader, res_path, id2speaker, dataset="train"):
    print(f"Generating GTA mel for {dataset} set...")

    model.eval()
    count_bugs = 0
    count_mels = 0
    for batchs in dataloader:
        for batch in batchs:
            spk = id2speaker[batch[2][0]]

            batch = to_device(batch, device)

            # Forward
            try:
                with torch.no_grad():
                    output = model(*(batch[2:]))
            except:
                count_bugs += 1
                logging.warn(f"Bug in runtime: {batch[0]}")
                logging.warn(f"Count bug: {count_bugs}")
                continue
            
            if batch[6].shape == output[1].shape:
                spk_path = os.path.join(res_path, "gta_mel", spk)
                if not os.path.isdir(spk_path):
                    os.makedirs(spk_path)
                path = os.path.join(spk_path, batch[0][0] + ".npy")
                count_mels += 1
                np.save(path, output[1].cpu().permute(0, 2, 1).detach().numpy())
    print("Count mels:", count_mels)

def main(args, configs):
    print("Prepare generating GTA mel ...")

    preprocess_config, model_config, train_config = configs
    os.makedirs(os.path.join(train_config["path"]["result_path"], "gta_mel"), exist_ok=True)

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    id2speaker = dict((v, k) for k, v in dataset.speaker_map.items())

    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )


    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    print("Number of FastSpeech2 Parameters:", num_param)

    gen_GTA_mel(model, dataloader, train_config["path"]["result_path"], id2speaker, dataset="train")


    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config["optimizer"]["num_workers"],
        collate_fn=dataset.collate_fn,
    )
    gen_GTA_mel(model, dataloader, train_config["path"]["result_path"], id2speaker, dataset="val")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "-d", "--data_path", type=str, required=False, help="path to preprocessed data"
    )
    parser.add_argument(
        "-l", "--lexicon_path", type=str, required=False, help="path to lexicon"
    )
    parser.add_argument(
        "-c", "--ckpt_path", type=str, required=False, help="path to ckpt"
    )
    parser.add_argument(
        "-r", "--result_path", type=str, required=False, help="path to result"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    if args.data_path is not None:
        preprocess_config["path"]["preprocessed_path"] = args.data_path
    if args.lexicon_path is not None:
        preprocess_config["path"]["lexicon_path"] = args.lexicon_path
    if args.ckpt_path is not None:
        train_config["path"]["ckpt_path"] = args.ckpt_path
    if args.result_path is not None:
        train_config["path"]["result_path"] = args.result_path
    # batch_size must be 1
    train_config["optimizer"]["batch_size"] = 1
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
