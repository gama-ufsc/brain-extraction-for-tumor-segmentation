import os

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())  # this has to be done before importing from nnunet

from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


if __name__ == '__main__':
    nnUNet_preprocessed = Path(os.environ['nnUNet_preprocessed'])
    results_dir = Path(os.environ['RESULTS_FOLDER'])

    for task_name in ['Task102_BraTS2020', 'Task107_TCGA_manual', 'Task108_TCGA_DICOM_nobe']:
        for model in ['2d', '3d_fullres']:
            plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
            trainer_class = get_default_configuration(model, task_name,
                                                      'nnUNetTrainerV2',
                                                      'nnUNetPlansv2.1')

            trainer = nnUNetTrainerV2(
                plans_file,
                'all',
                output_folder=output_folder_name,
                dataset_directory=dataset_directory,
                batch_dice=batch_dice,
                stage=stage,
                unpack_data=True,
                deterministic=False,
                fp16=True
            )
            if task_name == 'Task102_BraTS2020':
                if model == '2d':
                    trainer.max_num_epochs = 150
                else:
                    trainer.max_num_epochs = 500
            else:
                if model == '2d':
                    trainer.max_num_epochs = 100
                else:
                    trainer.max_num_epochs = 300
            trainer.max_num_epochs = 15  # TODO: remove epochs limit
            trainer.initialize(True)

            # TODO: implement continued training

            trainer.run_training()

            trainer.network.eval()
            trainer.validate(
                save_softmax=False,
                validation_folder_name='validation_raw',
                run_postprocessing_on_folds=True,
                overwrite=True
            )
