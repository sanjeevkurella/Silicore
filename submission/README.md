# AI-Based Navigation Error Recovery for Wafer Inspection Using Multi-Scale Edge-Assisted Template Matching

**Applied Materials Drift-Sense Hackathon – SEMICON India 2026**

---

## Overview

This project addresses the **AI-Based Navigation Error Recovery for Wafer Inspection** problem. Given a high-resolution semiconductor **reference image** and a **search image**, the objective is to accurately locate the reference pattern within the search image despite SEM imaging degradations such as:

- Scale variation
- Small rotational offsets
- SEM noise
- Blur
- Repetitive DRAM and FinFET structures

The proposed solution is a **classical computer vision pipeline** that combines multi-scale template matching with edge-assisted correlation and local refinement.

---

# Features

- Multi-scale template matching
- Gray + Edge score fusion
- Top-5 candidate selection
- Local refinement
- Rotation refinement (±2°)
- Confidence gap analysis
- CPU-only implementation
- No deep learning required

---

# Methodology

The localization pipeline consists of the following stages:

```
Reference Image
        │
        ▼
Grayscale Conversion
        │
        ▼
Canny Edge Detection
        │
        ▼
Multi-Scale Template Matching
(Scales 9.0 – 11.0)
        │
        ▼
Gray + Edge Score Fusion
(0.7 × Gray + 0.3 × Edge)
        │
        ▼
Top-5 Candidate Selection
        │
        ▼
Local Search Refinement
(40-pixel window)
        │
        ▼
Rotation Refinement
(-2°, -1°, 0°, +1°, +2°)
        │
        ▼
Final Localization
(x, y)
```

---

# Score Fusion

The combined matching score is computed as

```
Combined Score =
0.7 × Gray Correlation
+
0.3 × Edge Correlation
```

This improves robustness against SEM noise while preserving structural information.

---

# Repository Structure

```
baseline_solution/
│
├── infer_v4.py
├── evaluate_v2.py
├── run.py
├── README.md
├── requirements.txt
├── results.csv
└── prediction.png

output/
│
├── train/
│   ├── reference/
│   ├── search/
│   └── manifest.csv
```

---

# Requirements

- Python 3.10+
- OpenCV
- NumPy
- Pandas
- Matplotlib

Install dependencies using

```bash
pip install -r requirements.txt
```

---

# Running the Localization

## Single Image

```bash
python infer_v4.py --reference reference.png --search search.png
```

Output

- Predicted center coordinates
- Matching score
- Runtime
- prediction.png

---

## Dataset Evaluation

```bash
python evaluate_v2.py
```

Outputs

- results.csv
- Mean error
- Median error
- Worst error
- Accuracy statistics
- Average runtime

---

## Competition Submission

```bash
python run.py <input-dir> <output-dir>
```

---

# Experimental Setup

Dataset

- 20 benchmark image pairs

Image Resolution

- Reference: 100 × 100 pixels
- Search: 1000 × 1000 pixels

Scale Search

```
9.00
9.25
9.50
9.75
10.00
10.25
10.50
10.75
11.00
```

Rotation Search

```
−2°
−1°
0°
+1°
+2°
```

---

# Performance

| Metric | Result |
|---------|--------|
| Mean Error | 61.457 px |
| Median Error | 1.044 px |
| Worst Error | 651.814 px |
| Accuracy (≤1 px) | 45% |
| Accuracy (≤2 px) | 80% |
| Accuracy (≤4 px) | 80% |
| Accuracy (≤5 px) | 80% |
| Average Runtime | 1.308 s |

---

# Advantages

- Robust to SEM image degradations
- Handles moderate scale changes
- Supports small rotational drift
- Fast CPU execution
- Lightweight classical vision pipeline
- No model training required

---

# Limitations

- Performance degrades on highly repetitive DRAM layouts.
- Does not perform sub-pixel interpolation.
- No feature descriptor verification.
- Sensitive to large rotations beyond ±2°.
- No learning-based feature extraction.

---

# Future Work

- ORB/SIFT feature verification
- Adaptive confidence thresholding
- Sub-pixel localization refinement
- CNN-based feature extraction
- Vision Transformer matching
- LoFTR/SuperPoint integration

---

# Authors

Applied Materials Drift-Sense Hackathon Team

SEMICON India 2026

---

# License

This project was developed solely for the Applied Materials Drift-Sense Hackathon and is intended for academic and research purposes.
