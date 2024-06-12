
# IKEA-Manuals-at-Work

IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos

## Installation

1. Create a new conda environment:
   ```bash
   conda create -n IKEAVideo python=3.8
   ```

2. Activate the environment:
   ```bash
   conda activate IKEAVideo
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set the `PYTHONPATH` environment variable to include the `src` directory:
   ```bash
   export PYTHONPATH="./src:$PYTHONPATH"
   ```

## Data

- [Annotations](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/data/data.json)

## Dataset

The IKEA-Manuals-at-Work dataset provides detailed annotations for aligning 3D models, instructional manuals, and real-world assembly videos. It includes:

- 3D models of furniture parts
- Instructional manuals
- Assembly videos from the Internet
- Annotations:
  - Temporal step alignments
  - Temporal substep alignments
  - 2D-3D part correspondences
  - Part segmentations
  - Part 6D poses
  - Estimated camera parameters

For more information about the dataset, please refer to the [datasheet](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/datasheet.md).

<!-- ## Code Structure

- `src/`: Contains the source code for data loading, processing, and visualization.
  - `IKEAVideo/dataloader/`: Data loading utilities.
  - `IKEAVideo/utils/`: Utility functions for transformations and visualization.
- `data/`: Contains the annotation file and other data files.
- `notebooks/`: Jupyter notebooks for data exploration and visualization.
  - `data_viz.ipynb`: Notebook for loading and visualizing data from the dataset.
- `requirements.txt`: Lists the required Python dependencies.
- `README.md`: This file, providing an overview of the repository.
- `datasheet.md`: Detailed information about the dataset. -->

## Usage

1. Install the required dependencies and set up the environment as described in the [Installation](#installation) section.

2. Download the dataset files and place them in the appropriate directories:
  - Annotation file: [`data/data.json`](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/data/data.json)
  - 3D models: `data/parts/`
  - Instructional manuals: `data/manual_img/` and `data/pdfs/` 

3. Download the assembly videos from the provided Google Drive link: [IKEA Assembly Videos](https://drive.google.com/drive/folders/1x0mzse3WJUXSJ9MfeX1kvmApIfWsCGZw)
  - Download the videos from the Google Drive link and place them in the `data/videos/` directory.

4. Explore the dataset using the provided Jupyter notebook:
  - Open the [`notebooks/data_viz.ipynb`](https://github.com/yunongLiu1/IKEA-Manuals-at-Work/blob/main/notebooks/data_viz.ipynb) notebook.
  - Follow the instructions in the notebook to load and visualize the data.

## Citation

If you use this dataset in your research, please cite the following paper:

```bibtex
@inproceedings{liu2024ikea,
  title={IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos},
  author={Liu, Yunong and Liu, Weiyu and Khanna, Shubh and Eyzaguirre, Cristobal and Li, Manling and Niebles, Juan Carlos and Ravi, Vineeth and Mishra, Saumitra and Wu, Jiajun},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}
```

## Contact

For questions or feedback, please open an issue on the [GitHub repository](https://github.com/yunongLiu1/IKEA-Manuals-at-Work) or contact the authors directly.

