# HairNet: Fine-Grained Hair Type Classification  
DSAN 6600 — Neural Networks Final Project  

**Authors:** Morgan Dreiss · Viviana Luccioli · Satomi Ito · Yashwanth Devabathini  

---

## Abstract  
Hair type identification is essential for personalized hair care, yet current methods rely on subjective self-assessment and inconsistent visual comparison charts. No publicly available dataset or deep learning system exists for the full **Andre Walker 10-class hair typing framework**, which remains the most widely used taxonomy in industry settings.

To address this gap, we introduce **HairNet**, a fine-grained hair type classifier built using **EfficientNetV2-M** with transfer learning and **CORN ordinal regression** to model the natural progression from Type 1 through Types 4a–4c. We constructed a custom dataset of ~14,000 images through a multi-stage pipeline integrating SerpAPI retrieval, YOLOv8 person filtering, Keras augmentation, and MediaPipe hair segmentation, followed by extensive manual re-labeling.

Our model achieved **53.7% top-1 accuracy** and **88.7% within-one-class accuracy**, demonstrating strong performance on broad curl-pattern categories while highlighting challenges in fine-grained distinctions—especially among Type 4 subtypes. Results indicate that **data quality**, rather than model depth, is the primary bottleneck. HairNet represents the first end-to-end deep learning system for full 10-class hair type classification and establishes a baseline for future research.

📄 **Final report (PDF):** [`final_paper/final_paper.pdf`](final_paper/final_paper.pdf)  
🌐 **Deployed demo:** https://huggingface.co/spaces/med2106/hairNet  
📦 **Final model weights:** `code/EfficientNet_models/V2-M/efficientnet_v2m_FINALMODEL.pth`

---

## Repository Structure (Major Components)

```
final_project/
├── application/
│   └── streamlit_app/              # Deployed demo code
├── code/
│   ├── datacollection/
│   │   ├── serpapi/
│   │   │   └── serpapi_call.py     # SerpAPI image collection template
│   │   ├── data_augmentation/
│   │   │   └── data_aug_keras.py   # Keras augmentation pipeline
│   │   └── yolo/
│   │       └── yolo.py             # YOLOv8 filtering script
│   ├── image_processing/
│   │   └── hair_segmenter.py       # Hair segmentation via MediaPipe
│   ├── EfficientNet_models/
│   │   └── V2-M/
│   │       └── efficientnet_v2m_FINALMODEL.pth    # Final model
└── final_paper/
    └── final_paper.pdf             # Full written report
```


---

## Installation

### 1. Install `uv` (recommended)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv install
uv sync
source .venv/bin/activate
```


**Keywords**: Hair classification, CNN, EfficientNetV2-M, ordinal regression, CORN, computer vision, Andre Walker hair typing system

**Affiliation**: Georgetown University — Data Science & Analytics M.S. Program