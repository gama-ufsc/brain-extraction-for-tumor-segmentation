from pathlib import Path

from nnunet.inference.predict import predict_from_folder


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = Path(project_dir, 'data')
    models_dir = project_dir/'models'

    trainer = 'nnUNetTrainerV2'

    be_method = 'NoBE'

    test_data_dir = data_dir/'processed'/'Test'/'X'/be_method
    assert test_data_dir.exists()

    model = '2d'
    task = 'Task102_BraTS2020'

    model_dir = models_dir/'nnUNet'/model/task/'nnUNetTrainerV2__nnUNetPlansv2.1'
    assert model_dir.exists()

    preds_dir = data_dir/'predictions'/task/be_method
    preds_dir.mkdir(parents=True, exist_ok=True)

    predict_from_folder(str(model_dir), str(test_data_dir), str(preds_dir),
                        'all', False, 6, 2, None, 0, 1, True,
                        overwrite_existing=True, mode='normal',
                        overwrite_all_in_gpu=None, mixed_precision=True,
                        step_size=.5, checkpoint_name='model_final_checkpoint',)