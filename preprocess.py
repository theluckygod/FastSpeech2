import argparse

import yaml

from preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("--indir", type=str, required=False, help="path to input")
    parser.add_argument("--outdir", type=str, required=False, help="path to output")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if args.indir is not None:
        config["path"]["raw_path"] = args.indir
    if args.outdir is not None:
        config["path"]["preprocessed_path"] = args.outdir
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
