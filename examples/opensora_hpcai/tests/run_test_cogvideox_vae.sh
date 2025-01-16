#!/bin/sh
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
EXAMPLE_DIR="$(dirname "${PROJECT_DIR}")"
PACKAGE_DIR="$(dirname "${EXAMPLE_DIR}")"

export PYTHONPATH="${PROJECT_DIR}:${PACKAGE_DIR}:${PYTHONPATH}"

echo "Test VAE Encoder..."
mkdir -p tests/tmp
python tests/test_cogvideox_vae.py --mode 0
python tests/test_cogvideox_vae.py --mode 1
python tests/test_cogvideox_vae.py --compare
rm -rf tests/tmp
echo "Test VAE Encoder...Done"

echo "Test full VAE..."
mkdir -p tests/tmp
python tests/test_cogvideox_vae.py --mode 0 --full
python tests/test_cogvideox_vae.py --mode 1 --full
python tests/test_cogvideox_vae.py --compare
rm -rf tests/tmp
echo "Test full VAE...Done"
