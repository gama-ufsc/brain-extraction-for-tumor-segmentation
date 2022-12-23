from pathlib import Path

import numpy as np
import pandas as pd

from brats.utils import compute_all_scores


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = project_dir/'data'
    predictions_dir = data_dir/'predictions'

    labels_dir = data_dir/'processed/Test/y'
    labels_fpaths = list(labels_dir.glob('*.nii.gz'))
    labels_fpaths = sorted(labels_fpaths)

    sids = np.array([fp.name.split('_')[0] for fp in labels_fpaths])

    dfs = list()
    for task_be_pred_dir in predictions_dir.glob('*/*/*'):
        be_method = task_be_pred_dir.name
        model = task_be_pred_dir.parent.name
        task = task_be_pred_dir.parent.parent.name

        preds_fpaths = list(task_be_pred_dir.glob('*.nii.gz'))
        preds_fpaths = sorted(preds_fpaths)

        scores = compute_all_scores(preds_fpaths, labels_fpaths)
        scores['Dice'] = np.array(scores['Dice']).mean(axis=-1)
        scores['HD95'] = np.array(scores['HD95']).mean(axis=-1)

        df = pd.DataFrame(scores)
        df['SID'] = sids
        df['Model'] = model
        df['Task'] = task
        df['BE'] = be_method

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv('scores.csv')
