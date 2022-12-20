import os

from joblib import cpu_count
from pathlib import Path
from shutil import copy

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())  # this has to be done before importing from nnunet

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.experiment_planner_baseline_2DUNet_v21 import ExperimentPlanner2D_v21
from nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from nnunet.experiment_planning.utils import crop
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity


if __name__ == '__main__':

    nnUNet_preprocessed = Path(os.environ['nnUNet_preprocessed'])
    nnUNet_raw_data = Path(os.environ['nnUNet_raw_data_base'])/'nnUNet_raw_data'
    nnUNet_cropped_data = Path(os.environ['nnUNet_raw_data_base'])/'nnUNet_cropped_data'
    nnUNet_cropped_data.mkdir(exist_ok=True)

    n_threads = int(cpu_count() / 2)

    for task_name in ['Task102_BraTS2020', 'Task107_TCGA_manual', 'Task108_TCGA_nobe',]:
        # things here are a frozen version of what happens through nnUNet_plan_and_preprocess

        task_dir = nnUNet_raw_data/task_name
        assert task_dir.exists(), "task folder was not created"

        print('Preprocessing', task_name)

        verify_dataset_integrity(str(task_dir))

        crop(task_name, False, n_threads)
        cropped_task_dir = nnUNet_cropped_data/task_name
        assert cropped_task_dir.exists(), "something wrong happened during cropping"

        dataset_json = load_json(str(cropped_task_dir/'dataset.json'))
        modalities = list(dataset_json["modality"].values())
        collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        dataset_analyzer = DatasetAnalyzer(cropped_task_dir, overwrite=False, num_processes=n_threads)  # this class creates the fingerprint
        _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

        preprocessed_task_dir = nnUNet_preprocessed/task_name
        preprocessed_task_dir.mkdir(exist_ok=True)
        copy(cropped_task_dir/'dataset_properties.pkl', preprocessed_task_dir)
        copy(task_dir/'dataset.json', preprocessed_task_dir)

        planner3d = ExperimentPlanner3D_v21(str(cropped_task_dir), str(preprocessed_task_dir))
        planner3d.plan_experiment()
        planner3d.run_preprocessing((n_threads, n_threads))

        planner2d = ExperimentPlanner2D_v21(str(cropped_task_dir), str(preprocessed_task_dir))
        planner2d.plan_experiment()
        planner2d.run_preprocessing((n_threads, n_threads))
