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

This dataset was created by researchers from the CogAI group at Stanford University <span style="color:red">and researchers from J.P. Morgan AI Research.</span>

### Who funded the creation of the dataset? 

<span style="color:red">This project was funded by JPMorgan.</span>

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

No, the dataset focuses on furniture objects and assembly instructions. The remaining questions in this section are not applicable.
### Does the dataset identify any subpopulations (e.g., by age, gender)?

N/A

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

N/A

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

N/A

### Any other comments?

## Collection process

_\[T\]he answers to questions here may provide information that allow others to
reconstruct the dataset without access to it._

### How was the data associated with each instance acquired?

3D furniture models were collected from the IKEA-Manual dataset and associated assembly videos were collected from the IAW dataset. Segmentation masks, 6D part poses, step alignment, substep alignment are manually annotated.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

Detailed annotation procedures are described for annotating:
- Temporal segmentation and part identities
- Segmentation masks using an interactive tool powered by the SAM model
- 2D-3D correspondences and 6D part poses using PnP, RANSAC and manual refinement

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

<span style="color:red">The paper mentions that manual annotations were performed but does not provide details on who the annotators were or how they were compensated.</span>

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

### Over what timeframe was the data collected?

<span style="color:red">The timeframe of data collection is not specified.</span>

### Were any ethical review processes conducted (e.g., by an institutional review board)?

<span style="color:red">The paper does not mention any ethical review processes.</span>

### Does the dataset relate to people?

No, the dataset does not relate to people directly. The remaining questions in this section are not applicable.
### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

### Were the individuals in question notified about the data collection?
N/A

### Did the individuals in question consent to the collection and use of their data?
N/A

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?
N/A

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?
N/A

### Any other comments?

## Preprocessing/cleaning/labeling

_The questions in this section are intended to provide dataset consumers with the information
they need to determine whether the “raw” data has been processed in ways that are compatible
with their chosen tasks. For example, text that has been converted into a “bag-of-words” is
not suitable for tasks involving word order._
### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

Yes, the data was extensively annotated with:
- Temporal step alignments
- Temporal substep alignments  
- 2D-3D part correspondences
- Part segmentation masks
- Part 6D poses 
- Estimated camera parameters

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The raw assembly videos and manuals are retained along with the annotations.

### Is the software used to preprocess/clean/label the instances available?

<span style="color:red">The paper describes the annotation tools used (e.g. based on SAM for segmentation) but does not provide the code.</span>

### Any other comments?

None.

## Uses

### Has the dataset been used for any tasks already?

Yes, the paper demonstrates the use of the dataset for assembly plan generation, part segmentation, pose estimation, and part assembly based on videos.

### Is there a repository that links to any or all papers or systems that use the dataset?

<span style="color:red">The paper does not mention such a repository.</span>

### What (other) tasks could the dataset be used for?

The dataset could potentially be used for tasks such as:
- Grounding natural language instructions to videos
- Action recognition and localization
- 3D object tracking in videos
- Embodied AI for assembly

### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

The dataset involves IKEA's intellectual property. Commercial use may be restricted without permission.

### Are there tasks for which the dataset should not be used?

The dataset should not be used for building commercial furniture assembly systems without permission from IKEA, as the data involves their intellectual property.

### Any other comments?

None.

## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

<span style="color:red">The paper does not specify any plans for distribution.</span>

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

<span style="color:red">The distribution mechanism is not specified.</span>

### When will the dataset be distributed?

<span style="color:red">The paper does not indicate any timeline for dataset release.</span>

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

<span style="color:red">The licensing and terms of use are not specified in the paper.</span>

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

<span style="color:red">The dataset contains IKEA's furniture designs and assembly instructions, which are likely protected intellectual property. However, specific restrictions are not discussed in the paper.</span>

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

<span style="color:red">The paper does not mention any export controls or regulatory restrictions.</span>

### Any other comments?

None.

## Maintenance

### Who is supporting/hosting/maintaining the dataset?

<span style="color:red">Maintenance details are not provided.</span>

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

<span style="color:red">Contact information for the dataset owners is not provided in the paper.</span>

### Is there an erratum?

<span style="color:red">No erratum is mentioned in the paper.</span>

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

<span style="color:red">Plans for future dataset updates are not indicated.</span>

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?


The dataset does not relate to people directly, so this question is not applicable.

### Will older versions of the dataset continue to be supported/hosted/maintained?

<span style="color:red">Information about version maintenance is not provided as the dataset has not been released yet.</span>

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

<span style="color:red">The paper does not describe any mechanism for external contributions to the dataset.</span>

### Any other comments?

None.