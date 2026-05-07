# CiliaIO: Machine learning reveal spatial patterns of cilia beating dynamics in the zebrafish spinal cord

## Overview
CiliaIO is a state-of-the-art ML-based quantification methodology for cilia mophodynamics. This repository contains a complete multi-modal pipeline for quantifying ciliary motion from confocal microscopy videos. This workflow relies on single-shot cilia detection using a fine-tuned YOLO v11 model which identifies bounding box regions of interest (ROI) to pass to Meta's Segment Anything (SAM) vision transformer models, removing the need for computationally expensive iterative masking and tracking. Unlike prior work which use traditional computer vision approaches like binary thresholding and Canny detection, CiliaIO identifies where cilia are, enabling high quality masks even when cilia overlap. The goal of this project is to enable fast, robust, automated analysis of cilia dynamics overtime across diverse experimental conditions. CiliaIO can detect subtle changes in cilia morphodynamics that may not be detected by the human eye, but are pivotal to developing our understanding of ciliary function overall.

<div align="center">
  <img src="img/wt_13_f_4dpf_raw.PNG" alt="Raw Image" width="600">
  <p><b>Raw Frame:</b> 4-day-post-fertilization zebrafish cilia in the central canal raw confocal image.</p>

  <img src="img/wt_13_f_4dpf_cilia_io_output.PNG" alt="Cilia Skeleton" width="600">
  <p><b>CiliaIO Frame:</b> High quality detection, segmentation, skeletonization, and quantification of cilia.</p>
</div>

## Authors and Contact Information
**Authors:** Ece Atayeter<sup>1,2</sup>, Jason Ho<sup>3</sup>, Talon G. Blottin<sup>2</sup>, Ilyena B. Joe<sup>2</sup>, Ron Sistrunk<sup>2</sup>, Bo Zhang<sup>4</sup>, Lilianna Solnica Krezel<sup>4</sup>, Andreas Gerstlauer<sup>3</sup>, John B. Wallingford<sup>1</sup>, and Ryan S. Gray<sup>2,5</sup> 

<sup>1</sup>Department of Molecular Biosciences, The University of Texas at Austin, Austin, Texas 78712, USA. \
<sup>2</sup>Department of Nutrition and Pediatrics, Dell Pediatric Research Institute, The University of Texas at Austin, Austin, TX 78723, USA. \
<sup>3</sup>Chandra Family Department of Electrical and Computer Engineering, The University of Texas at Austin, Austin, Texas 78712, USA.\
<sup>4</sup>Department of Developmental Biology, Washington University School of Medicine, S Euclid Avenue, St. Louis, MO 63110, USA. \
<sup>5</sup>**Lead Contact Email:** ryan.gray@austin.utexas.edu \
**Direct all coding questions to:** jason_ho@utexas.edu

## CiliaIO's Core Pipeline

1. **Detection**  
   Cilia are detected frame-by-frame using a fine-tuned **YOLOv11m** model from Ultralytics.

2. **Tracking**  
   Object tracks are initialized and maintained using **ByteTrack** with additional logic for center-distance matching and overlap-based ID merging. Fallback heuristics handle missed detections and ensure temporal ID consistency across frames.

3. **Segmentation**
    Boxes are segmented using the Segment-Anything Model (SAM).

4. **Skeletonization**  
   Segmentation masks are skeletonized using the morphological thinning algorithm, which iteratively removes pixels on each side of the mask until one pixel remains.

5. **Quantification**  
   Tracked trajectories are  analyzed to extract per-cilium motion features, including spatial amplitude, beating frequency, beating regularity, centroid frequency, and more. Summary statistics are aggregated across videos for downstream analysis.

6. **Visualization**  
   The pipeline supports trajectory overlays, segmentation mask overlays on the raw confocal videos, skeleton tracking, and bounding box tracking.

## Directory Structure
- **data/**: This is where the data goes
- **sam/**: If coming from github, download the SAM ViT-H checkpoint 4b8939.pth file from Meta
- **src/**: Contains all of the source file pertaining to CiliaIO'
  - **cilia_io.py**: The top main file for the entire repository.
  - **ml_helpers.py**: Helper functions used by CiliaIO for machine learning.
  - **quantification_help.py**: Helper functions used by CiliaIO for quantification.
  - **benchmarking_cilia_tools.py**: Benchmarking cilia_io versus binary thresholding, canny thresholding, and pixel-based frequency segmentation

- **src/analysis/**: Contains all of the source files pertaining to downstream analysis with cilia_io
  - **analyze_csv_data_violin.py**: Uses the cilia_io output CSVs to generate Welch t-tests, 3D graphics, Cohen's power analysis, and violin plots.
  - **analyze_k_fold_cross_val_plots.py**: Contains utiltiies for coalescing k-fold cross validation plots together
  - **benchmark_segment_anything.py**: Benchmarks zero-shot segment-anything on manual labeled data.
  - **compare_csv_for_accuracy.py**: Script used for calculating the MAPE between manual and CiliaIO quantification outputs.
  - **visualize_skeleton_over_time.py**: Script to take skeleton pickles out of cilia_io output and produce .PNG temporal waveform plots
  
- **src/dataset_creation**: Create datasets for YOLO fine-tuning and k-fold cross validation
  - **create_yolo_training_dataset.py**: From raw label-studio projects, generates the combined train, validation, and test splits for the fine-tuning of the YOLO model.
  - **create_k_fold_cross_val_dataset.py**: Generate 3-fold stratified cross validation dataset for YOLO validation analysis
  
- **yolo/**: Contains the original Ultralytics Yolov11m model, the train and test scripts, as well as the final, fine-tuned model.
  - **configs.yaml**: YAML file for configuring the fine-tuning of the YOLO model.
  - **save_trainig_curves.py**: Saves training curves from the YOLO training flow as .csv files
  - **yolo_test.py**: Runs the test inference on the trained best YOLO model.
  - **yolo_train.py**: Runs training and validation curves for 300 epochs on the dataset.
  - **yolo11m.pt**: The actual pre-trained YOLOv11m model from Ultralytics.
  - **k_fold_cross_val.py**: Runs k_fold cross validation using the dataset created by '../src/dataset_creation/create_k_fold_cross_val_dataset.py'
  
## Dependencies
### NOTE: A GPU IS REQUIRED TO RUN CiliaIO

**See Requirements.txt for the most up to date dependencies from Python. All results were run with Python 3.10** \
To install the requirements, we recommend a virtual environment to keep your Python package installation different than the ones required by CiliaIO. You can do this as follows: 

```bash
python3 -m venv cilia_io_venv
source cilia_io_venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This will take some time because it installs all the PyTorch and CUDA binaries. Hang tight and grab a coffee.

## How to Run CiliaIO
See the user guide, titled: cilia_io_user_guide.pdf!