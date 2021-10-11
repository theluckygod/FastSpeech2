import argparse

import yaml
import os

from preprocessor import ljspeech, aishell3, libritts, my_data, vlsp_2021


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "my_data" in config["dataset"]:
        my_data.prepare_align(config)
    if "vlsp_2021" in config["dataset"]:
        vlsp_2021.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["corpus_path"] = args.indir
    if args.outdir is not None:
        config["path"]["raw_path"] = args.outdir
    main(config)
