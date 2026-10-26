"""Brain tumour segmentation pipeline.

Input: four co-registered, skull-stripped MRI sequences (T1, T1ce, T2, FLAIR) at 1 mm,
as distributed by BraTS. Model: MONAI brats_mri_segmentation (SegResNet, BraTS 2018),
outputs tumour core (TC), whole tumour (WT) and enhancing tumour (ET).

Outputs written to the study folder:
  vol_<seq>.bin   uint8 volume per sequence, RAS-ordered, gzip
  labels.bin      uint8 label map (1 necrotic/non-enhancing core, 2 oedema, 4 enhancing), gzip
  truth.bin       expert label map when a *_seg file was provided, gzip
  meshes.bin      brain + tumour sub-region surfaces, gzip
  report.json     measurements, location and model metadata
"""
import gzip
import json
import struct
import time
from pathlib import Path

import fast_simplification
import nibabel as nib
import numpy as np
import torch
import trimesh
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import SegResNet
from scipy import ndimage
from scipy.spatial import ConvexHull
from skimage import measure

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "brats_segresnet.pt"
SEQS = ["t1ce", "t1", "t2", "flair"]  # channel order the model was trained with

_model = None
_atlas = None


def model():
    global _model
    if _model is None:
        net = SegResNet(blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16, in_channels=4, out_channels=3, dropout_prob=0.2)
        net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        net.eval()
        _model = net
    return _model


def normalise(x):
    out = np.zeros_like(x, dtype=np.float32)
    m = x != 0
    if m.any():
        v = x[m]
        out[m] = (v - v.mean()) / (v.std() + 1e-8)
    return out


def segment(arrays, progress=lambda s: None):
    """arrays: dict seq -> float32 array in native orientation. Returns uint8 label map."""
    progress("Normalising intensities")
    x = np.stack([normalise(arrays[s].astype(np.float32)) for s in SEQS])[None]
    progress("Running SegResNet")
    inferer = SlidingWindowInferer(roi_size=(240, 240, 160), sw_batch_size=1, overlap=0.5)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        pred = torch.sigmoid(inferer(torch.from_numpy(x), model()))[0].numpy() > 0.5
    tc, wt, et = pred[0], pred[1], pred[2]
    lab = np.zeros(tc.shape, np.uint8)
    lab[wt] = 2
    lab[tc] = 1
    lab[et] = 4
    return lab


# ---------------------------------------------------------------------------- atlas
DK_LOBE = {
    "superiorfrontal": "Frontal", "rostralmiddlefrontal": "Frontal", "caudalmiddlefrontal": "Frontal", "parsopercularis": "Frontal",
    "parstriangularis": "Frontal", "parsorbitalis": "Frontal", "lateralorbitofrontal": "Frontal", "medialorbitofrontal": "Frontal",
    "precentral": "Frontal", "paracentral": "Frontal", "frontalpole": "Frontal",
    "superiorparietal": "Parietal", "inferiorparietal": "Parietal", "supramarginal": "Parietal", "postcentral": "Parietal", "precuneus": "Parietal",
    "superiortemporal": "Temporal", "middletemporal": "Temporal", "inferiortemporal": "Temporal", "bankssts": "Temporal", "fusiform": "Temporal",
    "transversetemporal": "Temporal", "entorhinal": "Temporal", "temporalpole": "Temporal", "parahippocampal": "Temporal",
    "lateraloccipital": "Occipital", "lingual": "Occipital", "cuneus": "Occipital", "pericalcarine": "Occipital",
    "rostralanteriorcingulate": "Cingulate", "caudalanteriorcingulate": "Cingulate", "posteriorcingulate": "Cingulate", "isthmuscingulate": "Cingulate",
    "insula": "Insula",
}
DK_NAME = {
    "superiorfrontal": "Superior frontal gyrus", "rostralmiddlefrontal": "Middle frontal gyrus (rostral)", "caudalmiddlefrontal": "Middle frontal gyrus (caudal)",
    "parsopercularis": "Pars opercularis (Broca)", "parstriangularis": "Pars triangularis (Broca)", "parsorbitalis": "Pars orbitalis",
    "lateralorbitofrontal": "Lateral orbitofrontal cortex", "medialorbitofrontal": "Medial orbitofrontal cortex", "precentral": "Precentral gyrus (primary motor)",
    "paracentral": "Paracentral lobule", "frontalpole": "Frontal pole", "superiorparietal": "Superior parietal lobule", "inferiorparietal": "Inferior parietal (angular)",
    "supramarginal": "Supramarginal gyrus", "postcentral": "Postcentral gyrus (primary sensory)", "precuneus": "Precuneus", "superiortemporal": "Superior temporal gyrus",
    "middletemporal": "Middle temporal gyrus", "inferiortemporal": "Inferior temporal gyrus", "bankssts": "Banks of superior temporal sulcus", "fusiform": "Fusiform gyrus",
    "transversetemporal": "Heschl's gyrus (primary auditory)", "entorhinal": "Entorhinal cortex", "temporalpole": "Temporal pole", "parahippocampal": "Parahippocampal gyrus",
    "lateraloccipital": "Lateral occipital cortex", "lingual": "Lingual gyrus", "cuneus": "Cuneus", "pericalcarine": "Pericalcarine cortex (primary visual)",
    "rostralanteriorcingulate": "Anterior cingulate (rostral)", "caudalanteriorcingulate": "Anterior cingulate (caudal)", "posteriorcingulate": "Posterior cingulate",
    "isthmuscingulate": "Cingulate isthmus", "insula": "Insula",
}
ASEG_NAME = {2: "Cerebral white matter", 41: "Cerebral white matter", 4: "Lateral ventricle", 43: "Lateral ventricle", 5: "Temporal horn",
             44: "Temporal horn", 10: "Thalamus", 49: "Thalamus", 11: "Caudate", 50: "Caudate", 12: "Putamen", 51: "Putamen", 13: "Globus pallidus",
             52: "Globus pallidus", 17: "Hippocampus", 53: "Hippocampus", 18: "Amygdala", 54: "Amygdala", 26: "Nucleus accumbens", 58: "Nucleus accumbens",
             28: "Ventral diencephalon", 60: "Ventral diencephalon", 16: "Brainstem", 7: "Cerebellar white matter", 46: "Cerebellar white matter",
             8: "Cerebellar cortex", 47: "Cerebellar cortex", 14: "Third ventricle", 15: "Fourth ventricle", 251: "Corpus callosum", 252: "Corpus callosum",
             253: "Corpus callosum", 254: "Corpus callosum", 255: "Corpus callosum"}
# regions whose damage carries high functional risk, with the hemisphere they matter in (None = both)
ELOQUENT = [
    ("Primary motor cortex", ["precentral", "paracentral"], None),
    ("Primary sensory cortex", ["postcentral"], None),
    ("Broca's area (speech production)", ["parsopercularis", "parstriangularis"], "lh"),
    ("Wernicke's region (language comprehension)", ["superiortemporal", "supramarginal", "bankssts"], "lh"),
    ("Primary visual cortex", ["pericalcarine"], None),
    ("Primary auditory cortex", ["transversetemporal"], None),
]


def atlas():
    """fsaverage aparc+aseg in MNI305, with lookup tables."""
    global _atlas
    if _atlas is None:
        img = nib.as_closest_canonical(nib.load(ROOT / "models" / "aparc+aseg.mgz"))
        lab = np.asarray(img.dataobj).astype(np.int32)
        bm = nib.as_closest_canonical(nib.load(ROOT / "models" / "fsaverage_brainmask.mgz"))
        mask = np.asarray(bm.dataobj) > 0
        names = {}
        dk = ["unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "corpuscallosum", "cuneus", "entorhinal", "fusiform",
              "inferiorparietal", "inferiortemporal", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual", "medialorbitofrontal",
              "middletemporal", "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine", "postcentral",
              "posteriorcingulate", "precentral", "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
              "superiortemporal", "supramarginal", "frontalpole", "temporalpole", "transversetemporal", "insula"]
        for i, n in enumerate(dk):
            names[1000 + i] = ("lh", n)
            names[2000 + i] = ("rh", n)
        _atlas = {"lab": lab, "aff": img.affine, "inv": np.linalg.inv(img.affine), "mask": mask, "names": names}
    return _atlas


def world_bbox(mask, aff):
    idx = np.argwhere(mask)
    lo, hi = idx.min(0), idx.max(0)
    corners = np.array([[lo[0], lo[1], lo[2], 1], [hi[0], hi[1], hi[2], 1]]).T
    w = (aff @ corners)[:3]
    return w.min(1), w.max(1)


# ---------------------------------------------------------------------------- measurements
def max_diameter_axial(mask):
    """Largest in-plane diameter and its perpendicular on the axial slice with most tumour (mm)."""
    areas = mask.sum((0, 1))
    if areas.max() == 0:
        return 0.0, 0.0, -1
    z = int(areas.argmax())
    pts = np.argwhere(mask[:, :, z]).astype(float)
    if len(pts) < 3:
        return 1.0, 1.0, z
    hull = pts[ConvexHull(pts).vertices]
    d = np.linalg.norm(hull[:, None] - hull[None], axis=2)
    i, j = np.unravel_index(d.argmax(), d.shape)
    axis = (hull[j] - hull[i]) / (d[i, j] + 1e-9)
    perp = np.array([-axis[1], axis[0]])
    proj = pts @ perp
    return float(d[i, j] + 1), float(proj.max() - proj.min() + 1), z


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def mesh_from_mask(mask, target, smooth=1.0, step=1):
    if mask.sum() < 20:
        return None
    m = ndimage.gaussian_filter(mask.astype(np.float32), smooth)
    v, f, _, _ = measure.marching_cubes(m, 0.5, step_size=step)
    tm = trimesh.Trimesh(v, f, process=True)
    trimesh.smoothing.filter_taubin(tm, iterations=10)
    if len(tm.faces) > target:
        vv, ff = fast_simplification.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int32), 1 - target / len(tm.faces))
        tm = trimesh.Trimesh(vv, ff, process=True)
    return tm.vertices.astype(np.float32), tm.faces.astype(np.uint32)


def to_u8(x, brain):
    v = x[brain]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    return np.clip((x - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8) * brain


def write_gz(path, arr_xyz):
    """Write an x,y,z array so that flat index = x + y*X + z*X*Y."""
    path.write_bytes(gzip.compress(np.ascontiguousarray(arr_xyz.transpose(2, 1, 0)).tobytes(), 6))


def run(study_dir: Path, files: dict, progress=lambda s: None):
    """files: seq -> path (t1, t1ce, t2, flair, optional seg)."""
    t0 = time.time()
    progress("Loading MRI sequences")
    imgs = {s: nib.load(str(files[s])) for s in SEQS}
    ref = imgs["t1"]
    for s in SEQS:
        if imgs[s].shape != ref.shape:
            raise ValueError(f"{s.upper()} is {imgs[s].shape}, T1 is {ref.shape}. All four sequences must be co-registered on the same grid.")
    zooms = ref.header.get_zooms()[:3]
    arrays = {s: np.asarray(imgs[s].dataobj, dtype=np.float32) for s in SEQS}
    lab = segment(arrays, progress)

    progress("Measuring the tumour")
    # reorient everything to RAS for display and measurement
    ornt = nib.orientations.io_orientation(ref.affine)
    ras = lambda a: nib.orientations.apply_orientation(a, ornt)
    ras_aff = ref.affine @ nib.orientations.inv_ornt_aff(ornt, ref.shape)
    lab_r = ras(lab).astype(np.uint8)
    vols = {s: ras(arrays[s]) for s in SEQS}
    brain = (vols["t1"] > 0) | (vols["flair"] > 0)
    brain = ndimage.binary_fill_holes(ndimage.binary_closing(brain, iterations=2))
    vox_ml = float(np.prod(zooms)) / 1000.0

    et, ncr, ed = lab_r == 4, lab_r == 1, lab_r == 2
    tc, wt = et | ncr, et | ncr | ed
    comps, ncomp = ndimage.label(wt)
    sizes = ndimage.sum(wt, comps, range(1, ncomp + 1)) * vox_ml if ncomp else []
    lesions = int(sum(1 for s in sizes if s >= 0.1))

    d1, d2, zslice = max_diameter_axial(tc if tc.any() else wt)
    report = {
        "detected": bool(wt.sum() * vox_ml >= 0.1),
        "volumes_ml": {"whole": round(wt.sum() * vox_ml, 2), "core": round(tc.sum() * vox_ml, 2), "enhancing": round(et.sum() * vox_ml, 2),
                       "necrotic": round(ncr.sum() * vox_ml, 2), "oedema": round(ed.sum() * vox_ml, 2), "brain": round(brain.sum() * vox_ml, 1)},
        "lesions": lesions,
        "diameters_mm": {"axial_max": round(d1, 1), "axial_perp": round(d2, 1), "slice": zslice},
        "shape": list(lab_r.shape), "voxel_mm": [float(z) for z in zooms],
    }
    if wt.any():
        idx = np.argwhere(wt)
        report["extent_mm"] = [int(x) for x in (idx.max(0) - idx.min(0) + 1)]
        report["centroid_vox"] = [round(float(x), 1) for x in idx.mean(0)]
        cidx = np.argwhere(tc if tc.any() else wt).mean(0)
        report["focus_vox"] = [int(round(x)) for x in cidx]
        v = report["volumes_ml"]
        report["composition"] = {k: round(v[k] / v["whole"] * 100, 1) if v["whole"] else 0 for k in ("enhancing", "necrotic", "oedema")}

        # location: per-axis linear map from the patient's brain box to the fsaverage brain box (MNI305)
        A = atlas()
        plo, phi = world_bbox(brain, ras_aff)
        alo, ahi = world_bbox(A["mask"], A["aff"])
        scale = (ahi - alo) / (phi - plo)
        to_mni = lambda w: alo + (w - plo) * scale
        world = (ras_aff @ np.c_[idx, np.ones(len(idx))].T)[:3].T
        mni = to_mni(world)
        report["centroid_mni"] = [round(float(x), 1) for x in mni.mean(0)]
        av = (A["inv"] @ np.c_[mni, np.ones(len(mni))].T)[:3].T.round().astype(int)
        av = np.clip(av, 0, np.array(A["lab"].shape) - 1)
        labels_hit = A["lab"][av[:, 0], av[:, 1], av[:, 2]]
        hemi_l = (mni[:, 0] < 0).mean()
        report["hemisphere"] = "Left" if hemi_l > 0.8 else "Right" if hemi_l < 0.2 else "Bilateral (crosses midline)"
        lobes, gyri, deep = {}, {}, {}
        for code, n in zip(*np.unique(labels_hit, return_counts=True)):
            code = int(code)
            if code in A["names"]:
                h, g = A["names"][code]
                if g in DK_LOBE:
                    lobes[DK_LOBE[g]] = lobes.get(DK_LOBE[g], 0) + n
                    key = f"{'Left' if h == 'lh' else 'Right'} {DK_NAME[g]}"
                    gyri[key] = gyri.get(key, 0) + n
            elif code in ASEG_NAME:
                deep[ASEG_NAME[code]] = deep.get(ASEG_NAME[code], 0) + n
        tot = len(labels_hit)
        cort = sum(lobes.values()) or 1
        report["lobes"] = [{"name": k, "pct": round(v / cort * 100, 1)} for k, v in sorted(lobes.items(), key=lambda x: -x[1])]
        report["gyri"] = [{"name": k, "pct": round(v / tot * 100, 1)} for k, v in sorted(gyri.items(), key=lambda x: -x[1])[:6]]
        report["deep"] = [{"name": k, "pct": round(v / tot * 100, 1)} for k, v in sorted(deep.items(), key=lambda x: -x[1]) if v / tot >= 0.01][:6]

        # distance from tumour core to eloquent cortex, measured in atlas space
        core_idx = np.argwhere(tc if tc.any() else wt)
        core_mni = to_mni((ras_aff @ np.c_[core_idx, np.ones(len(core_idx))].T)[:3].T)
        core_av = np.clip((A["inv"] @ np.c_[core_mni, np.ones(len(core_mni))].T)[:3].T.round().astype(int), 0, np.array(A["lab"].shape) - 1)
        core_mask = np.zeros(A["lab"].shape, bool)
        core_mask[core_av[:, 0], core_av[:, 1], core_av[:, 2]] = True
        dist = ndimage.distance_transform_edt(~core_mask)
        risks = []
        for title, regions, hemi in ELOQUENT:
            codes = [c for c, (h, g) in A["names"].items() if g in regions and (hemi is None or h == hemi)]
            m = np.isin(A["lab"], codes)
            if m.any():
                risks.append({"name": title, "mm": round(float(dist[m].min()), 1)})
        vent = np.isin(A["lab"], [4, 43, 5, 44])
        risks.append({"name": "Lateral ventricle", "mm": round(float(dist[vent].min()), 1)})
        cc = np.isin(A["lab"], [251, 252, 253, 254, 255])
        risks.append({"name": "Corpus callosum", "mm": round(float(dist[cc].min()), 1)})
        report["proximity"] = sorted(risks, key=lambda r: r["mm"])

    if files.get("seg"):
        truth = ras(np.asarray(nib.load(str(files["seg"])).dataobj).astype(np.uint8))
        report["validation"] = {
            "whole": round(dice(truth > 0, wt), 3),
            "core": round(dice((truth == 1) | (truth == 4), tc), 3),
            "enhancing": round(dice(truth == 4, et), 3),
        }
        write_gz(study_dir / "truth.bin", truth)

    progress("Building 3D surfaces")
    for s in SEQS:
        write_gz(study_dir / f"vol_{s}.bin", to_u8(vols[s], brain))
    write_gz(study_dir / "labels.bin", lab_r)
    meshes = [("brain", brain, 90000, 1.6, 2), ("oedema", wt, 30000, 0.9, 1), ("necrotic", ncr, 20000, 0.8, 1), ("enhancing", et, 25000, 0.7, 1)]
    mbuf, mmeta = bytearray(), []
    for name, m, target, sm, step in meshes:
        r = mesh_from_mask(m, target, sm, step)
        if r is None:
            continue
        v, f = r
        mmeta.append({"name": name, "nv": len(v), "nf": len(f)})
        mbuf += v.tobytes() + f.tobytes()
    (study_dir / "meshes.bin").write_bytes(gzip.compress(bytes(mbuf), 6))
    report["meshes"] = mmeta
    report["model"] = {"name": "SegResNet (MONAI brats_mri_segmentation)", "trained_on": "BraTS 2018, 285 glioma cases",
                       "reported_dice": {"core": 0.856, "whole": 0.903, "enhancing": 0.791}, "seconds": round(time.time() - t0, 1)}
    (study_dir / "report.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    import sys
    case = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    files = {s: next(case.glob(f"*_{s}.nii*")) for s in SEQS}
    seg = list(case.glob("*_seg.nii*"))
    if seg:
        files["seg"] = seg[0]
    rep = run(out, files, progress=lambda s: print("..", s, flush=True))
    print(json.dumps({k: rep[k] for k in rep if k not in ("meshes",)}, indent=1))
