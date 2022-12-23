# Brain Extraction Evaluation for Tumor Segmentation with Deep Learning

Accompanying the paper "Towards fully automated deep-learning-based brain tumor segmentation: is brain extraction still necessary?".

## Requirements

Our preprocessing pipeline is implemented using [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux), [CaPTk](https://cbica.github.io/CaPTk/Download.html) and [HD-BET](https://github.com/MIC-DKFZ/HD-BET), as detailed in the paper.

You will also need nnU-Net and our library of useful functions for brain tumor segmentation (brats), which are included as submodules in this repo.
I recommend that you install both in development mode (`python setup.py develop`) from the submodules, even though everything should work with the nnU-Net package installed from the [main source](https://github.com/MIC-DKFZ/nnUNet) (untested). 

## Data

We use data from BraTS 2020, which can be acquired in many different ways, and from [DICOM-Glioma-SEG](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=41517733), [BraTS-TCGA-LGG](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24282668) and [BraTS-TCGA-GBM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24282666).
Note, however, that the BraTS 2018 images of these last two sources are available only under request.

You should unzip BraTS2020 data into `data/raw/MICCAI_BraTS2020_TrainingData`, TCGA images in `data/raw/TCGA`, DICOM-Glioma-SEG in `data/raw/DICOM-Glioma-SEG` and DICOM_Glioma_SEG_Metadata in `data/raw/DICOM_Glioma_SEG_Metadata`.
You must also download the non-skull-stripped T1-weighted image of the [SRI24 atlas](https://www.nitrc.org/projects/sri24/) and store it as `data/raw/SRI24_T1.nii`.

The preprocessing expects the TCGA data to be organized in a slightly different way from what is provided directly in TCIA.
First, the TCGA folder inside `data/raw` must have two subdirectories, `LGG` and `GBM`, containing the respective data for each study.
Inside these, the structure is similar to what we get from TCIA, except all whitespaces are converted to underscores.
Unfortunately, this is necessary for some of the CLIs of the preprocessores we use. **This whitespace-to-underscores conversion is also required for DICOM-Glioma-SEG data**.
So, for example, the raw dicom files for the T1Gd modality of TCGA-02-0003 patient will be stored in the `data/raw/TCGA/GBM/TCGA-02-0003/13.000000-AX_T1_POST-19694/.` directory.

## BE Methods

As the BE methods require additional instalations, the current implementation applies only FSL's BET and HD-BET, as these installations are already required for the preprocessing pipeline.
However, this implementation suits any other BE method.
See the template for such in `make_test_data.py`, the `apply_XXX` function.

## Execution order

With the required packages (`environment.yml`) and the two submodules installed, run:

1. `make_dotenv.py` to create a `.env` file with the environment variables required by nnU-Net package
1. `make_nnunet_brats_task.py` and `make_nnunet_nobe_task.py` to preprocess the training data
1. `make_test_data.py` to preprocess the test data
1. `nnunet_pp_tasks.py` and `nnunet_train_models.py` to train the models
1. `nnunet_predict.py` to use the trained models for prediction
1. `compute_results.py` to generate a `scores.csv` file with Dice and HD95 of each image for all models and BE methods
