import subprocess

from setuptools import setup
import sys


def check_and_install_libomp():
    if sys.platform == "darwin":  # macOS
        try:
            subprocess.run(
                ["brew", "list", "libomp"],
                check=True,
                capture_output=True,
            )
            print("libomp is already installed.")
        except subprocess.CalledProcessError:
            print("libomp is not installed. Attempting to install it.")
            try:
                subprocess.run(["brew", "install", "libomp"], check=True)
                print("libomp installed successfully.")
            except subprocess.CalledProcessError as e:
                print("Failed to install libomp. Please install it manually.")
                sys.exit(1)


check_and_install_libomp()


def install_spacy():
    try:
        subprocess.run(
            ["pip", "install", "spacy"],
            check=True,
            # capture_output=True,
        )
        subprocess.run(
            ["python", "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
            # capture_output=True,
        )
        print("Installed spacy en_core_web_sm.")
    except subprocess.CalledProcessError:
        print("spacy en_core_web_sm could not be installed.")
        sys.exit(1)


install_spacy()
setup()
