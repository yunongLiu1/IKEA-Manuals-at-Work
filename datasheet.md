# Datasheet for dataset "IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos"
Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation

_The questions in this section are primarily intended to encourage dataset creators
to clearly articulate their reasons for creating the dataset and to promote transparency
about funding interests._

### For what purpose was the dataset created? 

The IKEA Video Manuals dataset was created to tackle the 4D grounding of assembly instructions in videos, which is essential for a holistic understanding of assembly in 3D space over time. Existing datasets have not provided such spatio-temporal alignments between 3D models, instructional manuals, and assembly videos.

### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

This dataset was created by researchers from the CogAI group at Stanford University and researchers from J.P. Morgan AI Research.

### Who funded the creation of the dataset? 

This project was funded by JPMorgan.

### Any other comments?

None.

## Composition

_Most of these questions are intended to provide dataset consumers with the
information they need to make informed decisions about using the dataset for
specific tasks. The answers to some of these questions reveal information
about compliance with the EU’s General Data Protection Regulation (GDPR) or
comparable regulations in other jurisdictions._

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

The dataset consists of 3D models of furniture parts, instructional manuals, assembly videos from the Internet, and annotations of dense spatio-temporal alignments between these data modalities.

### How many instances are there in total (of each type, if appropriate)?

The dataset provides 34,441 annotated video frames, aligning 36 IKEA manuals with 98 assembly videos for 6 furniture categories.

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The dataset is a curated sample of IKEA furniture assembly instances. It does not contain all possible IKEA furniture assembly examples.

### What data does each instance consist of? 

Each instance in the dataset consists of:
- 3D models of furniture parts
- Instructional manuals 
- Assembly videos
- Annotations including:
  - Temporal step alignments
  - Temporal substep alignments
  - 2D-3D part correspondences
  - Part segmentations
  - Part 6D poses
  - Estimated camera parameters

### Is there a label or target associated with each instance?

Yes, each video frame is annotated with temporal step alignments, temporal substep alignments, 2D-3D part correspondences, part segmentations, part 6D poses, and estimated camera parameters.

### Is any information missing from individual instances?

No, each instance contains complete information as described above.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

Yes, the dataset provides dense spatio-temporal alignments between the 3D models, instructional manuals, and assembly videos.

### Are there recommended data splits (e.g., training, development/validation, testing)?

<span style="color:red">The entire dataset is intended as a test set.</span>

### Are there any errors, sources of noise, or redundancies in the dataset?

The authors are not aware of errors or redundancies.

### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?


<span style="color:red">It is self-contained. The dataset relies on instructional manuals and assembly videos collected from the Internet. The authors provided a copy of the manuals and videos in Google Drive.</span>

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

No, the dataset does not contain any confidential information.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

No, the dataset only contains instructional assembly videos and related data which is not offensive in nature.

### Does the dataset relate to people? 

<span style="color:red">No, the dataset focuses on furniture objects and assembly instructions. The remaining questions in this section are not applicable.</span>
### Does the dataset identify any subpopulations (e.g., by age, gender)?

N/A

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

Our dataset does not provide any extra information about these people that was not already publicly available from their uploaded videos, and respects when videos are deleted/made private by virtue of using video ids.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

N/A


## Collection process

### How was the data associated with each instance acquired?

The IKEA Video Manuals dataset was created by identifying 36 IKEA objects from the IKEA-Manual dataset that have corresponding assembly videos in the IAW dataset. IKEA-Manual uses furniture names as identifiers, while IAW uses article numbers. To ensure correct correspondence, the unique IDs of the instruction manuals were matched.

The selected IAW videos capture real-world assembly processes with diverse backgrounds, viewpoints, and performer variability. This real-world complexity aligns with the goal of enabling the learning of robust assembly processes.

The 3D models in the dataset were collected from existing datasets and online repositories as part of the IKEA-Manual dataset. These models are segmented into individual parts that match the assembly manuals, enabling fine-grained reasoning about part-level assembly sequences.

The original assembly manuals were sourced from the official IKEA website and provide valuable ground truth information for evaluating the performance of assembly plan understanding algorithms.

### What mechanisms or procedures were used to collect the data?

The data was collected using the following mechanisms:
- IKEA objects were identified from the IKEA-Manual dataset and matched with corresponding assembly videos from the IAW dataset using the unique IDs of the instruction manuals.
- The 3D models were collected from existing datasets and online repositories as part of the IKEA-Manual dataset.
- The original assembly manuals were sourced from the official IKEA website.
- Detailed annotation procedures were used for annotating:
  - Temporal segmentation and part identities
  - Segmentation masks using an interactive tool powered by the SAM model
  - 2D-3D correspondences and 6D part poses using PnP, RANSAC and manual refinement
  - Temporal step and substep alignments

### If the dataset is a sample from a larger set, what was the sampling strategy?

The dataset is a curated sample of IKEA furniture assembly instances, selected based on the availability of corresponding assembly videos in the IAW dataset. The sampling strategy was deterministic, matching the unique IDs of the instruction manuals to ensure correct correspondence between the IKEA-Manual and IAW datasets.

### Who was involved in the data collection process and how were they compensated?

The authors of the paper were responsible for collecting the source data (videos, 3D models, and manuals), developing the annotation interface, and writing the annotation guidelines. 

<span style="color:red"> For the annotation process, the authors employed a team of annotators. The annotators were compensated for their work at a rate above the minimum wage. </span>

### Over what timeframe was the data collected?

The source data (videos, 3D models, and manuals) was collected over a period of 3 days in June 2023. The annotation process lasted until June 2024.


### Were any ethical review processes conducted?

N/A

### Does the dataset relate to people?

No, the dataset does not relate to people directly. The remaining questions in this section are not applicable.

## Preprocessing/cleaning/labeling

### Was any preprocessing/cleaning/labeling of the data done?

Yes, the data was extensively annotated with:
- Temporal step alignments
- Temporal substep alignments  
- 2D-3D part correspondences
- Part segmentation masks
- Part 6D poses 
- Estimated camera parameters

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

The raw assembly videos and manuals are retained along with the annotations.

### Is the software used to preprocess/clean/label the instances available?

Yes, we will release code for the interfaces we used for annotation upon acceptance.

### Any other comments?

None.

## Uses

### Has the dataset been used for any tasks already?

Yes, the paper demonstrates the use of the dataset for assembly plan generation, part segmentation, pose estimation, and part assembly based on videos.

### Is there a repository that links to any or all papers or systems that use the dataset?

Our paper is under review, but the data repository will provide links to papers that use the dataset in the future.

### What (other) tasks could the dataset be used for?
The dataset could potentially be used for tasks such as:

- 3D object tracking in videos
- Embodied AI for assembly
- Action recognition and localization


## Uses

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

The dataset has several potential biases and limitations, including:
1. Limited object diversity (IKEA furniture only)
2. Annotator subjectivity in labeling
3. Limited contextual information beyond furniture assembly

Future users should consider these factors when using the dataset and acknowledge them in their work.

### Are there tasks for which the dataset should not be used?

The dataset should not be used for training models that aim to generalize to diverse objects or settings, as it is primarily intended as a test set. 

### Any other comments?

None.

### Any other comments?

None.

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

The dataset will be freely available for public download.

### How will the dataset will be distributed?

The dataset is uploaded in a simple zip format on GitHub: https://github.com/yunongLiu1/IKEA-Manuals-at-Work

### When will the dataset be distributed?

The dataset is already publicly available for download.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset is released under the CC-BY-4.0 license.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

We build on IKEA-Manual and IAW, both are under CC. The additional annotations we provide does not involve IP-based restrictions.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

### Any other comments?


## Maintenance

### Who is supporting/hosting/maintaining the dataset?

The dataset is hosted on GitHub and will be maintained by the authors. The repository can be found at: https://github.com/yunongLiu1/IKEA-Manuals-at-Work

### How can the owner/curator/manager of the dataset be contacted?

The owners of the dataset can be contacted through the GitHub repository's issue tracker or by directly reaching out to the authors via the contact information provided in the repository.

### Is there an erratum?

Any erratum or updates to the dataset will be made available through the GitHub repository. Users should refer to the repository's README and release notes for any corrections or changes.

### Will the dataset be updated?

Yes, in the event that errors are found, the dataset will be uploaded as a new version at the same location.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

N/A

### Will older versions of the dataset continue to be supported/hosted/maintained?

Older versions of the dataset will be archived on the GitHub repository. 

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Researchers interested in extending or contributing to the dataset can submit pull requests to the GitHub repository. The authors will review and potentially integrate these contributions. Guidelines for contributing will be provided in the repository's documentation.

### Any other comments?

None.