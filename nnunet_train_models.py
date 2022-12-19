import os

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from nnunet.run.default_configuration import get_default_configuration
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    nnUNet_preprocessed = Path(os.environ['nnUNet_preprocessed'])
    results_dir = Path(os.environ['RESULTS_FOLDER'])

    # for task_name in ['Task102_BraTS2020', 'Task107_TCGA_bet', 'Task108_TCGA_DICOM_nobet']:
    for task_name in ['Task102_BraTS2020',]:
        plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration('2d', task_name,
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
        trainer.max_num_epochs = 150
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
