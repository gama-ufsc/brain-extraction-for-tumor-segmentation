# -*- coding: utf-8 -*-
"""Builds nnunet dataset with DICOM-Glioma-Seg images.
"""
import os
import pandas as pd
import pickle

from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

from brats.nnunet.datasets import make_nnunet_dataset


def build_nnunet_nobet(seg_list, data_dir, task_name):
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

    print('loading metadta')
    with open(data_dir/'train_val_metadata_df.pkl', 'rb') as f:
        metadata = pickle.load(f)

    brats_ids = pd.read_csv(data_dir/'MICCAI_BraTS2020_TrainingData/name_mapping.csv').dropna(axis=0, subset=['TCGA_TCIA_subject_ID']).set_index('TCGA_TCIA_subject_ID')['BraTS_2020_subject_ID']

    print('creating segmentation files and linking mri images')
    for seg_fpath in seg_list:
        dcm_mod = seg_fpath.parent.name
        sid = seg_fpath.parent.parent.name
        mod = dcm_mod.split('Glioma')[0][len('300.000000-'):][:-1]

        ref_fpath = next(data_dir.glob(f'TCGA/*/Pre*/{sid}/*t1*'))

        mod_img_metadata = metadata.loc[sid,dcm_mod]
        mod_label = mod_img_metadata['Modality']
        mod_dcm_dir = data_dir/mod_img_metadata['Filepath']

        brats_id = brats_ids.loc[sid]

        # TODO: WIP
        seg_dst_fpath = dst_dir/f"{brats_id}_{mod_label.lower()}_seg.nii.gz"
        mod_dst_fpath = dst_dir/f"{brats_id}_{mod_label.lower()}.nii.gz"
        if not seg_dst_fpath.exists():
            mod_fpath = dcm2nifti(mod_dcm_dir, '.tmpdir')

            mod_id = mod_dcm_dir.name.split('-')[-1]

            if sid.startswith('TCGA-FG-7643') and isinstance(mod_fpath, list) and mod_label == 'T1CE':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-HT-8114') and isinstance(mod_fpath, list) and mod_label == 'T1CE':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-CS-494') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-CS-5395') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-HT-768') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-HT-769') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-HT-785') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-HT-7882') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-HT-8105') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-DU-A6S6') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-7298') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-7014') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-7010') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-DU-7008') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-DU-6410') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-6401') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-6408') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-DU-640') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-DU-639') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-DU-58') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-06-0119') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-0128') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-013') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-014') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-0154') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-0158') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-0162') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-016') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-0177') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-017') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-018') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-0190') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-0213') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[0]
            if sid.startswith('TCGA-06-0240') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-1084') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-06-5408') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-27-183') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-27-2526') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]

            if sid.startswith('TCGA-76-4932') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-76-6191') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-76-6193') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
            if sid.startswith('TCGA-76-628') and isinstance(mod_fpath, list) and mod_label == 'T2':
                mod_fpath = mod_fpath[1]
                
            if sid == 'TCGA-06-0646' and mod_label == 'T1CE':
                # fix T1CE for CaPTk
                mod = nib.load(mod_fpath)
                fixed_mod = nib.Nifti1Image(mod.get_fdata()[:,:,:,0], mod.affine, mod.header)

                mod_fpath = str(mod_fpath).replace('.nii.gz', '_fixed.nii.gz')
                nib.save(fixed_mod, mod_fpath)
                mod_fpath = Path(mod_fpath)


        #     transform_fpath = next(metadata_dir.glob(f"*/{sid}/Registration_Transforms/*{mod.replace('_','__')}*.txt"))
            transform_fpath = next(metadata_dir.glob(f"*/{sid}/Registration_Transforms/*{mod_id}.txt"))

            mod_img = sitk.ReadImage(str(mod_fpath), imageIO="NiftiImageIO")
            seg_img = sitk.ReadImage(str(seg_fpath), imageIO="NiftiImageIO")
            ref_img = sitk.ReadImage(str(ref_fpath), imageIO="NiftiImageIO")

            transform = sitk.ReadTransform(str(transform_fpath))

            transf_mod_img = sitk.Resample(mod_img, referenceImage=ref_img, transform=transform.GetInverse())
            transf_seg_img = sitk.Resample(seg_img, referenceImage=ref_img, transform=transform.GetInverse())

            mod_tmp_fpath = '.tmpdir/mod.nii.gz'
            sitk.WriteImage(transf_mod_img, mod_tmp_fpath)

            reor = Reorient2Std()

            reor.inputs.in_file = mod_tmp_fpath
            reor.inputs.out_file = str(mod_dst_fpath)

            res = reor.run()

            seg_tmp_fpath = '.tmpdir/seg.nii.gz'
            sitk.WriteImage(transf_seg_img, seg_tmp_fpath)

            reor = Reorient2Std()

            reor.inputs.in_file = seg_tmp_fpath
            reor.inputs.out_file = str(seg_dst_fpath)

            res = reor.run()
            
            # TODO: !rm {mod_tmp_fpath}
            # TODO: !rm {seg_tmp_fpath}

    patient_names = None  # TODO

    print('creating dataset json')
    json_dict = OrderedDict()
    json_dict['name'] = "BraTS2020"
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
    data_dir = Path(project_dir, 'data', 'raw')

    seg_list = list((data_dir/'DICOM-Glioma-SEG').glob('**/seg.nii.gz'))
    build_nnunet_nobet(seg_list, data_dir, task_name='Task108_TCGA_DICOM_nobet')
