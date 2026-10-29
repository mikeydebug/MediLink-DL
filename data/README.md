# MediLink-DL Dataset Documentation

This directory contains data loaders, manifests, and cached medical images for the **MediLink-DL** research prototype.

## 1. Selected Clinical Dataset

- **Dataset**: COVID-19 Chest X-ray Dataset Initiative
- **Authors**: Joseph Paul Cohen, Paul Morrison, Lan Dao (Cohen et al., 2020)
- **Reference**: *COVID-19 Image Data Collection*, arXiv:2003.11597 ([GitHub repository](https://github.com/ieee8023/covid-chestxray-dataset))
- **Annotations**: COCO-format lung bounding boxes provided by `imageannotation.ai` (`annotations/imageannotation_ai_lung_bounding_boxes.json`)
- **Modality**: Posteroanterior (PA), Anteroposterior (AP), and AP Supine digital chest radiographs (CXRs).
- **Pathologies**: Severe viral pneumonia (COVID-19), SARS, ARDS, Pneumocystis pneumonia, bacterial pneumonia (*Streptococcus*, *Klebsiella*, *Legionella*), and healthy controls.
- **License**: Publicly accessible open-access repository for academic, scientific, and research use.

## 2. Directory Structure

```text
data/
├── README.md               # Dataset documentation and split governance (this file)
├── samples/                # Sample diagnostic images for smoke/unit testing
│   ├── sample_cxr.png
│   └── synthetic_phantom.png
└── real_cxr/               # Genuine chest X-ray collection (git-ignored image binaries)
    ├── annotations.json    # Original COCO lung bounding box annotations
    ├── manifest.csv        # Validated image-level metadata, patient IDs, and splits
    ├── splits.json         # Partition assignment indices (train, val, test)
    └── images/             # Raw radiograph JPEG/PNG files (~25 MB total)
        ├── auntminnie-a-2020_01_28_...jpeg
        └── ...
```

## 3. Patient-Level Splitting & Leakage Prevention

In medical imaging research, partitioning images randomly (image-level split) leads to severe **patient leakage** when multiple radiographs from the same individual appear across both training and evaluation subsets, artificially inflating diagnostic generalization metrics.

MediLink-DL implements strict **patient-level stratification**:
- Radiographs are grouped strictly by `patientid`.
- Partitions (70% Train, 15% Validation, 15% Test) are partitioned over distinct patient identifiers.
- **Verification Criterion**:
  $$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{val}} = \emptyset$$
  $$\text{Patients}_{\text{train}} \cap \text{Patients}_{\text{test}} = \emptyset$$
  $$\text{Patients}_{\text{val}} \cap \text{Patients}_{\text{test}} = \emptyset$$
- Exact split manifests are stored deterministically in `data/real_cxr/splits.json` using fixed seed `42`.

## 4. Bounding Box to Continuous ROI Mask Mapping

Ground-truth bounding boxes are recorded in COCO format:
$$[x, y, w, h]$$
During data loading (`src/dataset_real.py`):
1. The image is scaled from original resolution $(W_{\text{orig}}, H_{\text{orig}})$ to target tensor shape $(224, 224)$.
2. Bounding boxes are scaled proportionally:
   $$x_1 = \text{round}\left(x \times \frac{224}{W_{\text{orig}}}\right), \quad x_2 = \text{round}\left((x + w) \times \frac{224}{W_{\text{orig}}}\right)$$
   $$y_1 = \text{round}\left(y \times \frac{224}{H_{\text{orig}}}\right), \quad y_2 = \text{round}\left((y + h) \times \frac{224}{H_{\text{orig}}}\right)$$
3. A 2D target mask $M \in \{0, 1\}^{224 \times 224}$ is generated, representing bilateral lung and lesion parenchymal regions.

## 5. Automated Download & Verification Command

To download and validate the dataset locally:
```bash
./.venv/bin/python src/dataset_real.py
```
Or via the main CLI:
```bash
./.venv/bin/python src/main.py prepare-dataset
./.venv/bin/python src/main.py validate-dataset
```
