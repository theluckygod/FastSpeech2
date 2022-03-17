import argparse

import yaml
import time

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
    start_time = time.time()
    preprocessor.build_from_path()
    print(f"Processing time: {time.time() - start_time}")
