
import argparse
import os
import mlflow


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        help="path to image",
    )
    args = parser.parse_args()

    image_path = os.path.join(args.image, os.listdir(args.image)[0])

    mlflow.log_artifact(image_path)


if __name__ == "__main__":
    main()
