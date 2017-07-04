import argparse
from SingleDigitClassification.model import run_experiment

parser = argparse.ArgumentParser()

parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
)

parser.add_argument(
    '--data-path',
    help='GCS or local paths to data',
    nargs='+',
    required=True
)

args = parser.parse_args()
arguments = args.__dict__

job_dir = arguments.pop('job_dir')
data_path = arguments.pop('data_path')[0]

run_experiment(data_path, job_dir)
