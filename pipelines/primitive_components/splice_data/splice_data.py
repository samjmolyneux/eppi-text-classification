
import argparse
import json
import os
from pathlib import Path

from scipy.sparse import save_npz

from eppi_text_classification.utils import (
    load_csr_at_directory,
)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        help="path to data to be spliced",
    )
    parser.add_argument(
        "--num_rows",
        type=int,
        help="number of rows to keep",
    )
    parser.add_argument(
        "--spliced_data",
        type=str,
        help="path to spliced data",
    )
    args = parser.parse_args()

    data = load_csr_at_directory(args.data)
    num_rows = args.num_rows

    spliced_data = data[:num_rows]

    spliced_data_save_path = Path(args.spliced_data) / "splice_data.npz"
    save_npz(spliced_data_save_path, spliced_data)


if __name__ == "__main__":
    main()
