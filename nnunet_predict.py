from pathlib import Path

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())  # this has to be done before importing from nnunet

from nnunet.inference.predict import predict_from_folder


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = Path(project_dir, 'data')
    models_dir = project_dir/'models'

    for test_data_dir in data_dir.glob('processed/Test/X/*'):
        be_method = test_data_dir.name

        for model_dir in models_dir.glob('nnUNet/*/*/*'):
            task = model_dir.parent.name
            model = model_dir.parent.parent.name

            preds_dir = data_dir/'predictions'/task/be_method
            preds_dir.mkdir(parents=True, exist_ok=True)

            predict_from_folder(str(model_dir), str(test_data_dir),
                                str(preds_dir), 'all', False, 6, 2, None, 0, 1,
                                True, overwrite_existing=True, mode='normal',
                                overwrite_all_in_gpu=None, mixed_precision=True,
                                step_size=.5, checkpoint_name='model_final_checkpoint',)
