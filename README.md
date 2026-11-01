# Tumour Map

Local prototype: a doctor signs in, adds a patient, uploads a brain MRI and gets an AI tumour segmentation with a 3D view, MRI slices and a measurement report.

## Run

From the project folder:

```bash
python backend/server.py
```

Then open http://localhost:8210 and choose **Continue with the demo account**. The demo login is defined in `backend/seed.json`.

On first start the two bundled BraTS cases are segmented in the background (about 20 s each on a CPU).

## Requirements

Python 3.11+. Install dependencies with `pip install -r requirements.txt`.

## Input

Four NIfTI files per study: T1, T1 with contrast (T1ce), T2 and FLAIR. They must be skull-stripped, co-registered and at 1 mm, as in the BraTS protocol. Convert DICOM with `dcm2niix` first.

## How it works

- `backend/pipeline.py` runs the MONAI `brats_mri_segmentation` SegResNet (BraTS 2018), which outputs enhancing tumour, tumour core and whole tumour. It then measures volumes and diameters and estimates location against the FreeSurfer fsaverage atlas (MNI305).
- `backend/splat.py` is Part 2 of the BrainSplat-3D proposal. It fits 3D Gaussians to the tumour: they are placed inside the mask and optimised directly against the MRI volume, with occupancy matched to the mask and colour to T1ce intensity. No virtual cameras are used because the MRI is already 3D. It exports a standard 3DGS `.ply` (SH degree 0) plus fit metrics: Dice against the mask, splat volume and intensity PSNR. Re-run it on a processed study with `python backend/splat.py data/studies/<id>`.
- `backend/server.py` is a FastAPI app with login, patients, upload and a single-worker processing queue. Data is stored in `data/`, which is git-ignored.
- `web/index.html` is the client: patient list, upload, a 3D view (Three.js) and axial, coronal and sagittal slices with overlays.

## Online demo

A read-only build with the two sample cases already processed is at https://tumour-map.vercel.app. Rebuild it with `python tools/build_static.py`, which writes `site/`, then deploy that folder.

## Data and model credits

- Model: MONAI `brats_mri_segmentation` bundle, Apache 2.0 (`models/LICENSE-model.txt`).
- Sample scans: BraTS 2021 cases 00012 and 00014. Use of BraTS data requires citing Menze et al., IEEE TMI 2015, and Bakas et al., Sci Data 2017 (`models/data_license.txt`).
- Atlas: FreeSurfer fsaverage `aparc+aseg` (Desikan et al., 2006).

Research prototype, not a medical device.
