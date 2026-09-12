# InteractMed-VLM: Region-Aware, Patient-Friendly CXR Report Generation

**InteractMed-VLM** is a clinical multimodal AI pipeline for region-aware chest X-ray (CXR) report generation with independent triage assessment and visual explainability. The project extends the foundation established in *"Learning to Generate Clinically Coherent Chest X-Ray Reports"* (Lovelace & Mortazavi, EMNLP Findings 2020) by introducing anatomical region-guided conditioning, patient-accessible finding translation, safety-isolated urgency triage, and post-hoc visual attribution.

---

## 🔬 Pipeline Overview

```text
  [ Chest X-Ray Image ]
          │
          ▼
   Anatomical Selection (Heart / Lungs / Pleura)
          │
          ▼
   Region Grounding (Bounding Box + Regional Features)
          │
          ├───► Visual Explainability (Grad-CAM Heatmap + BBox Overlay)
          │
          ▼
   Clinical Finding Generation (Specialist-grade findings)
          │
          ├───► Patient-Friendly Translation (Accessible explanations)
          │
          ▼
   Independent Triage Classifier (Green / Yellow / Red Urgency Scoring)
```

1. **Vision Encoder**: Encodes the high-resolution CXR into spatially-dense visual representations.
2. **Region Grounding**: Resolves the user's anatomical selection (`heart`, `lungs`, `pleura`) into localized spatial coordinates (`[x, y, w, h]`) and pools regional feature embeddings.
3. **Report Generator**: Generates clinical finding sentences tailored to the grounded region, followed by plain-language patient explanations.
4. **Independent Triage Classifier**: Kept strictly isolated from report generation to ensure patient safety; classifies the clinical findings into **Green** (routine/normal), **Yellow** (non-emergent/follow-up), or **Red** (urgent/emergency).
5. **Visual Explainability**: Computes gradient-weighted activation maps (Grad-CAM) overlaid with bounding boxes to provide clear visual attribution for clinicians.
6. **Interactive Interface**: A Gradio web application for interactive human-in-the-loop CXR evaluation.

---

## 📁 Repository Structure

```text
InteractMed-VLM/
├── configs/
│   └── model_config.yaml         # Configuration file for models, backbones, and hyperparams
├── data/                         # Local datasets (gitignored)
│   ├── ms-cxr/                   # MS-CXR annotations (CSV and COCO JSON)
│   ├── chest-imagenome/          # Chest ImaGenome gold & silver datasets
│   └── mimic-cxr/                # Downloaded MIMIC-CXR-JPG images
├── manifests/                    # Processed manifests & download scripts
│   ├── unified_manifest.csv      # Unified dataset linking MS-CXR & Chest ImaGenome
│   └── download_list.txt         # Deduplicated GCS image paths for targeted download
├── notebooks/
│   └── exploratory_inspection.ipynb  # Interactive data exploration notebook
├── src/
│   ├── preprocessing/            # Manifest generation & dataset harmonization
│   │   └── build_manifest.py
│   ├── vision_encoder/           # Image encoding module (CXR -> visual tensor)
│   │   └── encoder.py
│   ├── region_grounding/         # Anatomical localization (image + region -> bbox + features)
│   │   └── grounding.py
│   ├── report_generator/         # Multimodal text generation & patient translation
│   │   └── generator.py
│   ├── triage/                   # Safety-isolated urgency classification (Green/Yellow/Red)
│   │   └── classifier.py
│   ├── explainability/           # Grad-CAM and bounding-box visual attribution
│   │   └── explainer.py
│   └── interface/                # Interactive Gradio demo application
│       └── app.py
├── tests/                        # Comprehensive unit tests with dummy fixtures
│   ├── test_build_manifest.py
│   ├── test_vision_encoder.py
│   ├── test_region_grounding.py
│   ├── test_report_generator.py
│   ├── test_triage.py
│   ├── test_explainability.py
│   └── test_interface.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Building the Unified Manifest
Generate the harmonized manifest and the GCS download list from MS-CXR and Chest ImaGenome Gold annotations:
```bash
python -m src.preprocessing.build_manifest \
    --ms-cxr-csv data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv \
    --ms-cxr-json data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.json \
    --gold-dir data/chest-imagenome/gold_dataset \
    --output-manifest manifests/unified_manifest.csv \
    --output-download-list manifests/download_list.txt
```

### 3. Downloading Required MIMIC-CXR Images
Only images referenced in the manifest need to be fetched:
```bash
# Example using gsutil
cat manifests/download_list.txt | gsutil -m cp -I data/mimic-cxr/
```

### 4. Running Unit Tests
All modules are tested on CPU using dummy tensor fixtures:
```bash
pytest tests/
# or
python -m unittest discover tests
```

### 5. Launching the Demo App
```bash
python -m src.interface.app
```
# InteractMed-VLM
