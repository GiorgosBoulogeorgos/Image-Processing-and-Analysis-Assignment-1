# Image Analysis and Processing — Assignment 1

This repository contains Assignment 1 from the **"Image Processing and Analysis"** course of the MSc Data Science & Information Technologies Master's programme (Bioinformatics — Biomedical Data Science Specialization) of the Informatics and Telecommunications department of the **National and Kapodistrian University of Athens (NKUA)**, under the supervision of professor **Vasileios Katsouros**, in the academic year **2025–2026**.

**Author:** Giorgos Boulogeorgos

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Exercises](#exercises)
   - [Exercise 1 — Step Transformation Function](#exercise-1--step-transformation-function)
   - [Exercise 2 — Enhancing a Dark Forest Image](#exercise-2--enhancing-a-dark-forest-image)
   - [Exercise 3 — Improving Perceived Brightness of a Pollen Image](#exercise-3--improving-perceived-brightness-of-a-pollen-image)
   - [Exercise 4 — Sharpening a Moon Photograph](#exercise-4--sharpening-a-moon-photograph)
   - [Exercise 5 — Reverse-Engineering an X-ray Processing Pipeline](#exercise-5--reverse-engineering-an-x-ray-processing-pipeline)
   - [Exercise 6 — Edges, Roof Angle, Corners and Windows](#exercise-6--edges-roof-angle-corners-and-windows)
   - [Exercise 7 — Billiard Cue: Angle, Rotation and Combination](#exercise-7--billiard-cue-angle-rotation-and-combination)
4. [Installation](#installation)
5. [Running the Notebook](#running-the-notebook)
6. [Image Assets](#image-assets)

---

## Overview

This assignment addresses **seven classical image processing problems** spanning point-wise intensity transformations, spatial filtering, contrast enhancement, edge and corner detection, geometric transformations, and the reverse-engineering of a multi-step processing pipeline. Every exercise is solved inside a single self-contained Jupyter notebook (`assignment_1_exercises.ipynb`), with each cell loading the input image, performing the processing, and displaying the relevant intermediate and final outputs together with their histograms or CDFs where appropriate.

The work is implemented in **Python 3** using the standard image-processing stack: OpenCV for the core image operations, scikit-image and SciPy for selected morphological and filtering primitives, scikit-learn (DBSCAN) for the unsupervised clustering used in window detection, and Matplotlib for all visualizations. Every methodological choice is justified in detail in the notebook's markdown cells: which transformation is appropriate, why a specific colour space (e.g. LAB or HSV) is used, why an adaptive method is preferred over a global one, and how each parameter is tuned with respect to the image's specific properties.

The emphasis throughout the assignment is on **explaining the choice of algorithm**, not just applying it. For each exercise the notebook discusses the structure of the image, the failure modes of simpler approaches, the underlying mathematical operation, the role of each tuning parameter, and the visual evidence in the output that confirms the chosen method works.

---

## Repository Structure

```
.
├── assignment_1_exercises.ipynb     # Main Jupyter notebook with all seven exercises
├── 1st_set_of_exercises_2026.docx   # Original assignment description
├── Documents/                       # Folder containing all input images
│   ├── 1000020113.jpg
│   ├── nature_dark_forest.jpg
│   ├── pollen-500x430px-96dpi.jpg
│   ├── First-photo-of-the-moon-from-Chandrayaan-2_ISRO.jpg
│   ├── image_1.jpg
│   ├── image_2.jpg
│   ├── image11.jpg
│   ├── image31.png
│   ├── image32.png
│   └── image33.png
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

The notebook expects all input images to live inside a folder named `Documents/` next to the notebook itself. This is set via the `IMAGE_DIR` constant at the top of the setup cell and can be edited if your images live elsewhere.

---

## Exercises

### Exercise 1 — Step Transformation Function

The task is to explain the impact of a given piecewise-constant intensity transformation on a grayscale image and verify it by applying it to a test image. The function maps the full 0–255 input range into only eight discrete output levels (10, 20, 50, 70, 100, 140, 180, 200). The transformation is implemented as a 256-entry Look-Up Table and applied via `cv2.LUT`. The result shows the four expected effects: **posterization** of smooth gradients into flat patches, **overall darkening** (the maximum output value is 200, not 255), **reduced dynamic range** from 256 to 8 distinct levels, and **loss of fine tonal gradations** within each input band. The output histogram exhibits exactly eight discrete spikes, one per LUT output level.

### Exercise 2 — Enhancing a Dark Forest Image

The task is to enhance a severely underexposed forest image in terms of perceived light and colour. A global gamma sweep (γ ∈ {0.5, 1.5, 2.5}) is shown first as a baseline and confirms the problem: any gamma strong enough to lift the shadows clips the already-bright sky to white. The chosen method is **CLAHE applied to the L channel of the LAB colour space**, with `clipLimit=3.0` and `tileGridSize=(8, 8)`. The result lifts the shadow detail in the forest floor and canopy while leaving the sky unclipped and preserving the original hue and saturation; the luminance histogram and CDF after CLAHE show a clear redistribution of intensities away from the dark end without any new pile-up at 255.

### Exercise 3 — Improving Perceived Brightness of a Pollen Image

The task is to improve the perceived brightness of a low-contrast pollen microscopy image so that the fine surface texture becomes visible. The notebook first verifies that the image is effectively grayscale (all three channels are identical), then compares three methods: gamma correction (γ = 1.5), global histogram equalization, and **CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)`**. CLAHE is selected as the final method because it lifts the dark regions while preserving the small intensity differences that encode the surface texture of the pollen grain — exactly the detail that global HEQ tends to merge away. The histogram + CDF comparison across the four versions makes the trade-offs visually explicit.

### Exercise 4 — Sharpening a Moon Photograph

The task is to sharpen a slightly soft photograph of the Moon taken by Chandrayaan-2. Two classical methods are compared: **Unsharp Masking (USM)** and **Laplacian / high-boost filtering**. An α-sweep over USM ({0.5, 2, 5, 10}, σ = 2 fixed) identifies **α = 2 as the operating point** that maximises perceived crispness of crater rims before bright/dark ringing halos appear at α ≥ 5. The Laplacian sweep over k ∈ {0.3, 0.7, 1.5} produces a sharper result but with visibly more amplified sensor noise in the flat lunar maria. USM is therefore preferred as it offers explicit scale control (via σ) and built-in noise suppression from its Gaussian blur stage.

### Exercise 5 — Reverse-Engineering an X-ray Processing Pipeline

The task has two parts: (a) guess the processing steps that turn the chest X-ray `image_1.jpg` into the very different-looking `image_2.jpg`, and (b) propose a pipeline that approximates `image_1` starting from `image_2`. By comparing histograms and CDFs the forward pipeline is identified as **gamma correction (γ = 0.2) → photometric negation (s = 255 − r) → 3×3 high-pass sharpening filter**, with each intermediate step matched visually and statistically against `image_2`. The reverse pipeline applies the inverse operations in reverse order — Gaussian blur, negation, gamma correction with γ = 2 — and recovers an image close to `image_1` in tone and structure, though not pixel-identical because the forward gamma compression and sharpening amplification both destroy information that cannot be perfectly recovered.

### Exercise 6 — Edges, Roof Angle, Corners and Windows

The task is to perform four sub-operations on `image11.jpg` (a wooden house with a turf roof): find the main edges, estimate the angle of the diagonal roof edges, find the corners, and use those corners to locate the windows. Edges are extracted with **Canny preceded by a Gaussian blur (σ = 1.4)**, where the high/low thresholds are derived automatically from **Otsu's method** on the smoothed image. The roof angle is estimated with the probabilistic Hough transform restricted to the upper 55% of the image, then taking the median angle per slope — yielding approximately **+43° (left slope) and −43° (right slope)**, giving an apex angle close to 90°. Corners are computed with both Harris and Shi-Tomasi (`goodFeaturesToTrack`, 300 points). The window-localization stage isolates the dark facade as the largest dark connected component, filters the Shi-Tomasi corners to those whose local 11×11 neighbourhood contains both bright (>160) and dark (<50) pixels, and clusters the survivors with **DBSCAN (`eps=30, min_samples=2`)**, drawing one bounding box per detected window.

### Exercise 7 — Billiard Cue: Angle, Rotation and Combination

The task is to (a) estimate the angle of the billiard cue in `image31.png` and verify it by rotating the image to match `image32.png`, and (b) combine `image31` and `image32` to produce a result similar to `image33.png`. The cue angle is found by running **`cv2.HoughLinesP` (with `minLineLength=120`)** on the Canny edge map and selecting the **longest detected segment** — the cue is the only object long and straight enough to produce such a segment. The image is then rotated by the negative of that angle around its centre via `cv2.warpAffine` with bilinear interpolation, producing a visual match to `image32.png`. For the combination step, the rotated image is converted to **HSV** and a mask built from the cue's distinctive hue range (8–20) AND saturation range (110–150) is applied to the Value channel, zeroing all non-cue pixels. The result reproduces the near-black appearance of `image33.png` with only the brightness of the cue retained.

---

## Installation

### Prerequisites

- **Python 3.10 or newer** (any modern Python 3.x should work; the project was developed on Python 3.10+).
- **Git** to clone the repository.
- Approximately 500 MB of free disk space for dependencies.

### Step 1 — Clone the repository

```bash
git clone https://github.com/GiorgosBoulogeorgos/Image-Processing-and-Analysis-Assignment-1.git
cd Image-Processing-and-Analysis-Assignment-1
```

### Step 2 — Create a virtual environment (recommended)

Using a virtual environment isolates the project's dependencies from your system Python.

On macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You should see `(venv)` prepended to your shell prompt while the environment is active. To deactivate it later, simply run `deactivate`.

### Step 3 — Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs NumPy, OpenCV, Matplotlib, scikit-image, SciPy, scikit-learn, and Jupyter — everything the notebook needs.

---

## Running the Notebook

After installation, launch Jupyter from the project root:

```bash
jupyter notebook assignment_1_exercises.ipynb
```

or, if you prefer the modern JupyterLab interface:

```bash
jupyter lab assignment_1_exercises.ipynb
```

Once the notebook opens in your browser:

1. **Run the setup cell first** (the very first code cell). This imports all libraries and defines the `show` helper used throughout the notebook. Note: the first cell also contains a `pip install` line that can be commented out after the first successful run if you already installed dependencies via `requirements.txt`.
2. **Each exercise cell is self-contained.** After running the setup, you can run any exercise independently — they do not depend on each other.
3. **Make sure the `Documents/` folder is present** next to the notebook with all the input images listed in the [Image Assets](#image-assets) section.

If the working directory differs from the project root, edit the `IMAGE_DIR` line in the setup cell to point to the correct image folder.

---

## Image Assets

The notebook expects the following images inside a folder named `Documents/`:


| Filename                                              | Used in    |
| ----------------------------------------------------- | ---------- |
| `1000020113.jpg`                                      | Exercise 1 |
| `nature_dark_forest.jpg`                              | Exercise 2 |
| `pollen-500x430px-96dpi.jpg`                          | Exercise 3 |
| `First-photo-of-the-moon-from-Chandrayaan-2_ISRO.jpg` | Exercise 4 |
| `image_1.jpg`, `image_2.jpg`                          | Exercise 5 |
| `image11.jpg`                                         | Exercise 6 |
| `image31.png`, `image32.png`, `image33.png`           | Exercise 7 |


All images except the test image for Exercise 1 are provided as part of the course material. The test image used for Exercise 1 can be replaced by any other grayscale image with a sufficiently wide intensity distribution — the LUT will be applied identically.
