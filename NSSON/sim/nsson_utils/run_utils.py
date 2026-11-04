import os
import csv
import datetime


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(root, prefix="run"):
    """
    root: e.g., '~/NSSON/shared/cnn_runs'
    returns: absolute path to run dir, e.g. '~/NSSON/shared/cnn_runs/run_20260528_162300'
    """
    root = os.path.expanduser(root)
    os.makedirs(root, exist_ok=True)
    run_id = f"{prefix}_{timestamp()}"
    run_dir = os.path.join(root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def csv_logger(csv_path, fieldnames):
    """
    Returns a callable log(row_dict) that appends to a CSV with given fieldnames.
    Creates file and header on first use.
    """
    csv_path = os.path.expanduser(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)

    def log(row):
        nonlocal file_exists
        write_header = not file_exists
        file_exists = True
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    return log
