# -*- coding: utf-8 -*-
"""Builds nnunet dataset with BraTS'2020 images.
"""
import os
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from brats.nnunet.datasets import make_nnunet_dataset


def build_nnunet_raw_brats(img_list, task_name):
    """From nnUNet's repo.
    """
    # load environment variables
    load_dotenv(find_dotenv())

    Path(os.environ['nnUNet_raw_data_base']).mkdir(exist_ok=True)
    Path(os.environ['nnUNet_preprocessed']).mkdir(exist_ok=True)
    Path(os.environ['RESULTS_FOLDER']).mkdir(exist_ok=True)

    nnUNet_raw_data = Path(os.environ['nnUNet_raw_data_base'], 'nnUNet_raw_data')
    nnUNet_raw_data.mkdir(exist_ok=True)

    # make base dirs
    target_base = Path(nnUNet_raw_data, task_name)
    target_imagesTr = Path(target_base, "imagesTr")
    target_imagesVal = Path(target_base, "imagesVal")
    target_imagesTs = Path(target_base, "imagesTs")
    target_labelsTr = Path(target_base, "labelsTr")

    target_base.mkdir(exist_ok=True)
    target_imagesTr.mkdir(exist_ok=True)
    target_imagesVal.mkdir(exist_ok=True)
    target_imagesTs.mkdir(exist_ok=True)
    target_labelsTr.mkdir(exist_ok=True)

    print('creating segmentation files and linking mri images')
    patient_names = make_nnunet_dataset(img_list, target_imagesTr,
                                        target_seg_dir=target_labelsTr)

    print('creating dataset json')
    json_dict = OrderedDict()
    json_dict['name'] = task_name[8:]
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see BraTS2020"
    json_dict['licence'] = "see BraTS2020 license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "FLAIR",
        "1": "T1",
        "2": "T1c",
        "3": "T2",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing",
        "3": "enhancing",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                            patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = Path(project_dir, 'data', 'raw', 'MICCAI_BraTS2020_TrainingData')

    brats_img_dirs = list(data_dir.iterdir())
    build_nnunet_raw_brats(brats_img_dirs, task_name='Task102_BraTS2020')

    brats_tcga_names = pd.read_csv(data_dir/'name_mapping.csv')
    brats_tcga_names = brats_tcga_names[~brats_tcga_names['TCGA_TCIA_subject_ID'].isna()]

    tcga_img_dirs = [img for img in brats_img_dirs
                     if img.name in brats_tcga_names['BraTS_2020_subject_ID'].values]

    build_nnunet_raw_brats(tcga_img_dirs, task_name='Task107_TCGA_manual')
