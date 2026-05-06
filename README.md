# DiSM-UNet

**Leveraging Foundation Model Priors for Robust Medical Image Segmentation via Global-Local Semantic Modulation**

DiSM-UNet is a generalized biomedical semantic-modulated dual-encoder framework for medical image segmentation. By coupling an STViT-based medical encoder with a pre-trained DINOv3 foundation encoder, the architecture effectively synthesizes local medical textures and global foundation semantics. This approach provides superior segmentation performance and excellent robustness, particularly for challenging structures with severe scale variation and boundary ambiguity.

## Architecture Overview

DiSM-UNet adopts an asymmetric dual-stream design:

* **Medical Expert Encoder (STViT):** Extracts multi-scale anatomical textures, local boundaries, and fine-grained structural cues.
* **Visual Foundation Encoder (DINOv3):** Provides global semantic priors and category-agnostic shape representations.
* **G-MFE Module:** A core fusion module featuring Global Geometric Affine Modulation (Global-GAM) and a Multi-Feature Extraction Block (MFEBlock) to adaptively align, modulate, and fuse heterogeneous features. 

## Evaluated Datasets

The model demonstrates strong generalization capabilities across multiple imaging modalities:

* **Synapse:** Abdominal multi-organ CT segmentation
* **ACDC:** Cardiac MRI multi-class segmentation
* **ISIC 2018:** Skin lesion segmentation from RGB dermoscopic images

## Getting Started

### Prerequisites

*(Add instructions for setting up the environment, e.g., `pip install -r requirements.txt`)*

### Training

*(Add the command to start training, e.g., `python train.py`)*

### Testing

*(Add the command to run inference, e.g., `python test.py`)*
