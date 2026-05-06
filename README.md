# DiSM-UNet

[cite_start]**Leveraging Foundation Model Priors for Robust Medical Image Segmentation via Global-Local Semantic Modulation** [cite: 1]

[cite_start]DiSM-UNet is a generalized biomedical semantic-modulated dual-encoder framework for medical image segmentation[cite: 12]. [cite_start]By coupling an STViT-based medical encoder with a pre-trained DINOv3 foundation encoder, the architecture effectively synthesizes local medical textures and global foundation semantics[cite: 13, 14]. [cite_start]This approach provides superior segmentation performance and excellent robustness, particularly for challenging structures with severe scale variation and boundary ambiguity[cite: 16, 40].

## Architecture Overview

[cite_start]DiSM-UNet adopts an asymmetric dual-stream design[cite: 83]:
* [cite_start]**Medical Expert Encoder (STViT):** Extracts multi-scale anatomical textures, local boundaries, and fine-grained structural cues[cite: 83].
* [cite_start]**Visual Foundation Encoder (DINOv3):** Provides global semantic priors and category-agnostic shape representations[cite: 84].
* [cite_start]**G-MFE Module:** A core fusion module featuring Global Geometric Affine Modulation (Global-GAM) and a Multi-Feature Extraction Block (MFEBlock) to adaptively align, modulate, and fuse heterogeneous features[cite: 85]. 

## Evaluated Datasets
[cite_start]The model demonstrates strong generalization capabilities across multiple imaging modalities[cite: 553, 554]:
* [cite_start]**Synapse:** Abdominal multi-organ CT segmentation [cite: 555]
* [cite_start]**ACDC:** Cardiac MRI multi-class segmentation [cite: 559]
* [cite_start]**ISIC 2018:** Skin lesion segmentation from RGB dermoscopic images [cite: 562]

## Getting Started

### Prerequisites
*(Add instructions for setting up the environment, e.g., `pip install -r requirements.txt`)*

### Training
*(Add the command to start training, e.g., `python train.py`)*

### Testing
*(Add the command to run inference, e.g., `python test.py`)*
