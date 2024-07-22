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


def install_spacy():
    try:
        subprocess.run(
            ["python3", "-m", "spacy download", "en_core_web_sm"],
            check=True,
            capture_output=True,
        )
        print("spacy is already installed.")
    except subprocess.CalledProcessError:
        print("spacy is not installed. Attempting to install it.")
        try:
            subprocess.run(["pip", "install", "spacy"], check=True)
            print("spacy installed successfully.")
        except subprocess.CalledProcessError as e:
            print("Failed to install spacy. Please install it manually.")
            sys.exit(1)


check_and_install_libomp()

setup()

install_spacy()
