<div align="center">

# IKEA Manuals at Work
### 4D Grounding of Assembly Instructions on Internet Videos
*NeurIPS 2024 Datasets and Benchmarks*

[Yunong Liu](http://yunongliu.com/)<sup>1</sup>, [Cristobal Eyzaguirre](https://ceyzaguirre4.github.io)<sup>1</sup>, [Manling Li](https://limanling.github.io)<sup>1</sup>, Shubh Khanna<sup>1</sup>, [Juan Carlos Niebles](https://www.niebles.net)<sup>1</sup>, Vineeth Ravi<sup>2</sup>, Saumitra Mishra<sup>2</sup>, [Weiyu Liu](http://weiyuliu.com)<sup>1*</sup>, [Jiajun Wu](https://jiajunwu.com)<sup>1*</sup>

<sup>1</sup>Stanford University &nbsp;&nbsp; <sup>2</sup>J.P. Morgan AI Research  
<sup>*</sup>Equal advising

[[Project Website]](https://yunongliu1.github.io/ikea-video-manual/) [[Paper]](https://arxiv.org/pdf/2411.11409) [[Dataset Setup Guide]](#dataset-setup) [[Notebook]](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/notebooks/data_viz.ipynb)

<img src="./assets/dataset_visualization.gif" width="600px"/>
</div>

## Overview
The IKEA-Manuals-at-Work dataset provides detailed annotations for aligning 3D models, instructional manuals, and real-world assembly videos. This is the first dataset to provide 4D grounding of assembly instructions on Internet videos, offering high-quality, spatial-temporal alignments between assembly instructions, 3D models, and real-world internet videos.

### Key Features
- 🪑 36 furniture models from 6 categories
- 🎥 98 assembly videos from the Internet
- 🔄 Dense spatio-temporal alignments between instructions and videos
- 📊 Rich annotations including part segmentation, 6D poses, and temporal alignments

## Getting Started

### Installation
```bash
# Create and activate conda environment
conda create -n IKEAVideo python=3.8
conda activate IKEAVideo

# Install dependencies
pip install -r requirements.txt

# Set PYTHONPATH
export PYTHONPATH="./src:$PYTHONPATH"
```

### Dataset Structure
```
data/
├── data.json           # Main annotation file
├── parts/             # 3D model files
├── manual_img/        # Instruction manual images
├── pdfs/              # Original PDF manuals
└── videos/            # Assembly videos
```

### Dataset Contents
The dataset includes:
- **3D Models**: Detailed 3D models of furniture parts
- **Instruction Manuals**: Step-by-step assembly instructions
- **Assembly Videos**: Real-world assembly videos from the Internet
- **Rich Annotations**:
   - ⏱️ Temporal step alignments
   - 🔄 Temporal substep alignments
   - 🎯 2D-3D part correspondences
   - 🎨 Part segmentations
   - 📐 Part 6D poses
   - 📷 Estimated camera parameters

For detailed information about the dataset, please refer to our [datasheet](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/datasheet.md).

### Dataset Setup
1. **Download Required Files**:
  - Annotation file: [`data/data.json`](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/data/data.json)
  - Assembly videos: [Stanford Digital Repository](https://purl.stanford.edu/sg200ps4374)
  - Clone the repo to obtain other resources (e.g. 3D models, manual images)
  - Place downloads in their respective directories as shown in Dataset Structure

2. **Explore the Dataset**:
  Check our tutorial notebook: [`notebooks/data_viz.ipynb`](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/notebooks/data_viz.ipynb)



## Applications
The dataset supports various research directions:
- 🔍 Assembly plan generation
- 🎯 Part-conditioned segmentation
- 📐 Part-conditioned pose estimation
- 🎥 Video object segmentation
- 🛠️ Shape assembly with instruction videos

## 3D Pose Refine Interface

This tool refines 3D object poses in video frames after initial pose estimation (e.g., via PnP). It allows users to make fine-tuned adjustments to object rotations and translations, visualize results in a 3D scene, and save refined poses.

### Before Running

Replace the following placeholders in `./annotation_tool/Pose-Refine-Interface-Release/server.py` and `./annotation_tool/Pose-Refine-Interface-Release/main.js` with your actual settings:

#### In `server.py`
- `OUTPUT_FOLDER = "OUTPUT_FOLDER_PLACEHOLDER"`: Output directory (e.g., `./output`).
- `OBJ_FOLDER = "OBJ_FOLDER_PLACEHOLDER"`: Path to OBJ files (e.g., `/path/to/parts`).
- `BASE_JSON_PATH = "BASE_JSON_PATH_PLACEHOLDER"`: Path to `data.json` (e.g., `./data.json`).
- `VIDEO_FOLDER = "VIDEO_FOLDER_PLACEHOLDER"`: Path to video files (e.g., `/path/to/video`).
- `BASE_PROGRESS_DATA_PATH = "BASE_PROGRESS_DATA_PATH_PLACEHOLDER"`: Path to user data (e.g., `/path/to/pose_reannotation_data/`).
- `port = "PORT_PLACEHOLDER"`: Server port (e.g., `5000`).
- `host = "HOST_PLACEHOLDER"`: Server host (e.g., `0.0.0.0`).

#### In `main.js`
- `const port = "PORT_PLACEHOLDER"`: Backend port (e.g., `5000`).
- `const host = "HOST_PLACEHOLDER:${port}"`: Backend URL (e.g., `http://localhost:5000`).

### How to Run
Under folder ./annotation_tool/Pose-Refine-Interface-Release/

1. **Install Dependencies**:
   - Frontend: `npm install` (installs `three`, `vite`, etc.; `node_modules` not included).
   - Backend: `pip install flask flask-cors trimesh numpy`.

2. **Start Frontend**:
   ```bash
   cd ./annotation_tool/Pose-Refine-Interface-Release
   npx vite --port 8000
   ```
  
3. **Start Backend**:
   ```bash
   cd ./annotation_tool/Pose-Refine-Interface-Release
   python server.py
   ```


## License
This dataset is released under the CC-BY-4.0 license.

## Citation
If you find this dataset useful for your research, please cite:
```bibtex
  @inproceedings{
  liu2024ikea,
  title={{IKEA} Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos},
  author={Yunong Liu and Cristobal Eyzaguirre and Manling Li and Shubh Khanna and Juan Carlos Niebles and Vineeth Ravi and Saumitra Mishra and Weiyu Liu and Jiajun Wu},
  booktitle={The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2024}
  }
```

## Contact
For questions and feedback:
- 📮 Open an issue on this [GitHub repository](https://github.com/yunongLiu1/IKEA-Manuals-at-Work)
- 📧 Email Yunong Liu
