#!/usr/bin/env python3

import argparse
import logging
import os
from pathlib import Path

from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--perturb-speed",
        type=bool,
        default=False,
        help="Whether to use speed perturbation.",
    )
    return parser

def preprocess_local_data(perturb_speed: bool = False):
    logging.info("Loading manifest")
    
    cuts_file = Path("data/manifests/cuts_train.jsonl.gz")
    if not cuts_file.is_file():
        logging.error(f"Cannot find manifest file: {cuts_file}")
        raise FileNotFoundError(f"Manifest file not found: {cuts_file}")

    cuts_train = CutSet.from_jsonl_lazy(cuts_file)

    logging.info("Compute fbank features")
    
    extractor = Fbank(FbankConfig(num_mel_bins=80))

    with LilcomChunkyWriter("data/fbank") as storage:
        if perturb_speed:
            logging.info("Applying speed perturbation")
            cuts_train = cuts_train + cuts_train.perturb_speed(0.9) + cuts_train.perturb_speed(1.1)
        
        cuts_train = cuts_train.compute_and_store_features(
            extractor=extractor,
            storage=storage,
            num_jobs=min(4, os.cpu_count()),
        )

    logging.info("Saving cuts with features")
    cuts_train.to_jsonl("data/fbank/cuts_train.jsonl.gz")

def main():
    args = get_parser().parse_args()
    preprocess_local_data(perturb_speed=args.perturb_speed)
    logging.info("Done")

if __name__ == "__main__":
    main()
