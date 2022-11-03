"""Create `.env` files suitable for nnU-Net.
"""
from pathlib import Path


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[0]

    with open('.env', 'w') as f:
        f.write(f"nnUNet_raw_data_base={project_root}/data/raw\n")
        f.write(f"nnUNet_preprocessed={project_root}/data/processed\n")
        f.write(f"RESULTS_FOLDER={project_root}/models\n")
