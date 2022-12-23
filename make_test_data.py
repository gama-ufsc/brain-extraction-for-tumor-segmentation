import shutil

from pathlib import Path
from typing import Dict

import nibabel as nib
import pandas as pd

from brats.nnunet.datasets import copy_BraTS_segmentation_and_convert_labels
from brats.preprocessing.preprocessing import PreprocessorHDBET, apply_mask_match_brats
from brats.preprocessing.captk_wrappers import greedy_apply_transforms, greedy_registration
from brats.preprocessing.hdbet_wrapper import hd_bet
from brats.preprocessing.nipype_wrappers import fsl_bet
from brats.utils import dcm2nifti
from nibabel.orientations import aff2axcodes, axcodes2ornt, apply_orientation, inv_ornt_aff


def apply_hd_bet_fast(modalities_fpaths: Dict[str,Path], tmpdir: Path):
    modality_fpath = modalities_fpaths['t1']

    _, brain_mask_fpath = hd_bet(
        str(modality_fpath),
        str(tmpdir/(modality_fpath.name.split('_')[0] + '_mask')),
        mode='fast',
    )

    return brain_mask_fpath

def apply_hd_bet(modalities_fpaths: Dict[str,Path], tmpdir: Path):
    modality_fpath = modalities_fpaths['t1']

    _, brain_mask_fpath = hd_bet(
        str(modality_fpath),
        str(tmpdir/(modality_fpath.name.split('_')[0] + '_mask')),
    )

    return brain_mask_fpath

def apply_bet(modalities_fpaths: Dict[str,Path], tmpdir: Path):
    modality_fpath = modalities_fpaths['t1']

    _, brain_mask_fpath = fsl_bet(
        str(modality_fpath),
        str(tmpdir/(modality_fpath.name.split('_')[0] + '_mask')),
        fast=True,
    )

    return brain_mask_fpath

def apply_bet_reduced_bias(modalities_fpaths: Dict[str,Path], tmpdir: Path):
    modality_fpath = modalities_fpaths['t1']

    _, brain_mask_fpath = fsl_bet(
        str(modality_fpath),
        str(tmpdir/(modality_fpath.name.split('_')[0] + '_mask')),
        fast=False,
    )

    return brain_mask_fpath

def apply_XXX(modalities_fpaths: Dict[str,Path], tmpdir: Path):
    """You can implement other BE methods following this template.
    
    The function must take all modalities as argument, generate a brain mask,
    store it at `tmpdir` and return the filepath of the mask. Then, add it to
    the dict of functions to be applied down below in the main.
    """
    raise NotImplementedError


if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parents[0]
    data_dir = Path(project_dir, 'data')

    tmpdir = Path('.tmpdir').resolve()
    tmpdir.mkdir(exist_ok=True)

    sri24_fpath = data_dir/'raw/SRI24_T1.nii'
    assert sri24_fpath.exists(), f"SRI24 T1 template not found in {sri24_fpath}"

    modality_number = {
        "flair": "0000",
        "t1": "0001",
        "t1ce": "0002",
        "t2": "0003",
    }

    metadata = pd.read_pickle(data_dir/'raw/test_metadata.pkl')
    assert all(metadata['Filepath'].apply(lambda r: (data_dir/'raw'/r).exists())), (
        "Missing Test set files"
    )

    name_mapping = pd.read_csv(data_dir/'raw/MICCAI_BraTS2020_TrainingData/name_mapping.csv')
    tcga2brats = name_mapping[~name_mapping['TCGA_TCIA_subject_ID'].isna()]
    tcga2brats = tcga2brats.set_index('TCGA_TCIA_subject_ID')['BraTS_2020_subject_ID']
    tcga2brats = dict(tcga2brats)

    # create directory for preprocessed images
    dst_dir = data_dir/'processed/Test'
    dst_dir.mkdir(exist_ok=True)
    (dst_dir/'X').mkdir(exist_ok=True)
    (dst_dir/'y').mkdir(exist_ok=True)

    # get modalities fpaths
    metadata = metadata.set_index(['Subject ID', 'Modality'])
    fpaths = metadata.groupby(level=0).apply(lambda df: df.xs(df.name).Filepath.to_dict()).to_dict()

    for tcga_id in list(fpaths.keys())[:5]:  # TODO: remove images limit
        # get fpath of segmentation file
        t1_fpath = fpaths[tcga_id]['T1']
        possible_segs = list(data_dir.glob(f"raw/TCGA/*/Pre*/{tcga_id}/*GlistrBoost*"))
        possible_segs = sorted(possible_segs)  # keeps `manually corrected` one (if any) last
        seg_fpath = possible_segs[-1]

        # convert dicom images to nifti
        t1_fpath = dcm2nifti(data_dir/'raw'/fpaths[tcga_id]['T1'], tmpdir)
        t1ce_fpath = dcm2nifti(data_dir/'raw'/fpaths[tcga_id]['T1CE'], tmpdir)
        flair_fpath = dcm2nifti(data_dir/'raw'/fpaths[tcga_id]['FLAIR'], tmpdir)
        t2_fpath = dcm2nifti(data_dir/'raw'/fpaths[tcga_id]['T2'], tmpdir)

        # several patches for series with multiple images
        if tcga_id.startswith('TCGA-FG-7643') and isinstance(t1ce_fpath, list) and len(t1ce_fpath) == 2:
            t1ce_fpath = t1ce_fpath[0]
        if tcga_id.startswith('TCGA-HT-8114') and isinstance(t1ce_fpath, list) and len(t1ce_fpath) == 2:
            t1ce_fpath = t1ce_fpath[1]

        if tcga_id.startswith('TCGA-CS-494') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-CS-5395') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        if tcga_id.startswith('TCGA-HT-768') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-HT-769') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-HT-785') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-HT-7882') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-HT-8105') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        if tcga_id.startswith('TCGA-DU-A6S6') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-7298') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-7014') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-7010') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-DU-7008') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-DU-6410') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-6401') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-6408') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-DU-640') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-DU-639') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-DU-58') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        if tcga_id.startswith('TCGA-06-0119') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-0128') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-013') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-014') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-0154') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-0158') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-0162') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-016') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-0177') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-017') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-018') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-0190') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-0213') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[0]
        if tcga_id.startswith('TCGA-06-0240') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-1084') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-06-5408') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        if tcga_id.startswith('TCGA-27-183') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-27-2526') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        if tcga_id.startswith('TCGA-76-4932') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-76-6191') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-76-6193') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]
        if tcga_id.startswith('TCGA-76-628') and isinstance(t2_fpath, list) and len(t2_fpath) == 2:
            t2_fpath = t2_fpath[1]

        # register modalities to template
        dpp = PreprocessorHDBET(
            template_fpath=sri24_fpath,
            tmpdir=tmpdir,
            bet_modality='T1',
            mode='fast',
            tta=0,
            registration='captk',
        )
        modalities_at_template, raw2template_transformations = dpp.run(
            flair_fpath=flair_fpath,
            t1_fpath=t1_fpath,
            t1ce_fpath=t1ce_fpath,
            t2_fpath=t2_fpath,
        )

        # register images at template to brats T1CE
        brats_t1ce_fpath = next(seg_fpath.parent.glob('*t1Gd.nii*'))
        template2brats_transformation = greedy_registration(
            brats_t1ce_fpath,
            modalities_at_template['t1ce'].get_filename(),
            str(tmpdir/'template_to_brats.mat'),
        )

        # apply transformations to the raw images
        modalities_registered = dict()
        for mod, mod_fpath in {'t1': t1_fpath, 't1ce': t1ce_fpath,
                               'flair': flair_fpath, 't2': t2_fpath}.items():
            mod_transforms = [template2brats_transformation, ] + raw2template_transformations[mod]
            modalities_registered[mod] = greedy_apply_transforms(
                str(mod_fpath),
                str(brats_t1ce_fpath),
                str(tmpdir/('brats_'+mod_fpath.name)),
                mod_transforms,
            )

        # correct affine matrix of images to the std orientation
        def fix_affine(image):
            ornt = axcodes2ornt(aff2axcodes(image.affine))
            image_data = apply_orientation(image.get_fdata(), ornt)

            new_affine = image.affine @ inv_ornt_aff(ornt, image_data.shape)

            return nib.Nifti1Image(image_data, new_affine, image.header)

        t1_image = fix_affine(nib.load(modalities_registered['t1']))
        t1ce_image = fix_affine(nib.load(modalities_registered['t1ce']))
        t2_image = fix_affine(nib.load(modalities_registered['t2']))
        flair_image = fix_affine(nib.load(modalities_registered['flair']))

        # store raw images (w/o skull-stripping)
        nobe_dir = dst_dir/'X/NoBE'
        nobe_dir.mkdir(exist_ok=True)

        registered_modalities = {
            m: nobe_dir/f"{tcga_id}_{modality_number[m]}.nii.gz"
            for m in ['t1', 't1ce', 't2', 'flair']
        }
        nib.save(t1_image, registered_modalities['t1'])
        nib.save(t1ce_image, registered_modalities['t1ce'])
        nib.save(t2_image, registered_modalities['t2'])
        nib.save(flair_image, registered_modalities['flair'])

        # fix segmentation labels
        fixed_seg_fpath = tmpdir/('fixed_'+seg_fpath.name)
        copy_BraTS_segmentation_and_convert_labels(str(seg_fpath),
                                                   str(fixed_seg_fpath))
        seg = nib.load(fixed_seg_fpath)
        seg = nib.Nifti1Image(seg.get_fdata().astype('uint8'),
                              seg.affine, seg.header)
        seg.set_data_dtype('uint8')  # store labels as uint to save disk
        seg = fix_affine(seg)

        nib.save(seg, dst_dir/'y'/f"{tcga_id}_seg.nii.gz")

        # apply BE methods
        be_methods = {
            'HD-BET-fast': apply_hd_bet_fast,
            'HD-BET': apply_hd_bet,
            'BET': apply_bet,
            'BET-reduced-bias': apply_bet_reduced_bias,
            # INSERT NEW METHOD HERE
            # 'new-be-method': apply_new_be_method
        }
        for be_method, apply_be_method in be_methods.items():
            brain_mask_fpath = apply_be_method(registered_modalities, tmpdir)
            brain_mask = nib.load(brain_mask_fpath)

            be_dir = dst_dir/f"X/{be_method}"
            be_dir.mkdir(exist_ok=True)
            for modality, modality_fpath in registered_modalities.items():
                # besides BE, also shifts histogram to match brats images
                brain_modality_fpath = apply_mask_match_brats(
                    modality_fpath,
                    brain_mask_fpath,
                    tmpdir/'brain_'
                )

                shutil.move(brain_modality_fpath, be_dir/modality_fpath.name)

        # clear temporary files
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(exist_ok=True)
