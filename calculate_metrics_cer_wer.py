import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    wers, cers = [], []
    for gt_file in Path(args.gt_dir).glob("*.txt"):
        pred_file = Path(args.pred_dir) / gt_file.name
        if pred_file.exists():
            gt = gt_file.read_text().strip().lower()
            pred = pred_file.read_text().strip().lower()
            wers.append(calc_wer(gt, pred))
            cers.append(calc_cer(gt, pred))
    print(f"WER: {sum(wers) / len(wers)}")
    print(f"CER: {sum(cers) / len(cers)}")


if __name__ == "__main__":
    main()
