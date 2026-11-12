# A Spatial-Spectral-Frequency Interactive Network for Multimodal Remote Sensing Classification
This is a code demo for the paper "A Spatial-Spectral-Frequency Interactive Network for Multimodal Remote Sensing Classification"

## Abstract: 

Deep learning-based methods have achieved significant success in remote sensing Earth observation data analysis. Numerous feature fusion techniques address multimodal remote sensing image classification by integrating global and local features. However, these techniques often struggle to extract structural and detail features from heterogeneous and redundant multimodal images, particularly in label-scarce scenarios. With the goal of introducing frequency domain learning to model key and sparse detail features, this paper introduces the spatial–spectral-frequency interaction network (S2Fin), which integrates pairwise fusion modules across the spatial, spectral, and frequency domains. Specifically, we propose a high-frequency enhancement and sparse
transformer to refine spectral signatures by adaptively enhancing discriminative high-frequency components. For spatial-frequency interaction, we present a depth-wise strategy: the adaptive frequency channel module fuses low-frequency structural information with enhanced details in shallow layers, while the high-frequency resonance mask amplifies modality-consistent regions in deep layers using phase similarity. In addition, a spatial–spectral attention fusion module bridges the gap between spectral and spatial branches at intermediate depths. Extensive experiments on four benchmark datasets demonstrate that S2Fin exhibits good robustness and generalization, and its performance significantly outperforms state-of-the-art methods in few-sample settings.

URL to the paper: https://www.sciencedirect.com/science/article/pii/S0031320326012744

## Requirements:
```bibtex
einops==0.8.0
mamba_ssm==2.2.4
matplotlib==3.9.4
numpy==2.0.2
opencv_python==4.11.0.86
opencv_python_headless==4.12.0.88
rasterio==1.4.3
scikit_learn==1.6.1
scipy==1.13.1
timm==1.0.13
torch==2.5.1
torchvision==0.20.1
tqdm==4.67.1
```

## Datasets:

The experiments are conducted on the following multimodal remote sensing datasets: https://drive.google.com/drive/folders/1lukFHjE2EYRdHPdreFxJ9c-egwbtZjqe?usp=drive_link

As for the Yellow River Estuary Dataset, please contact email at gaoyunhao@bit.edu.cn

## Citation
If you find this work useful in your research, please cite our paper:
```bibtex
@article{LIU2026114309,
  title={A spatial-spectral-frequency interactive network for multimodal remote sensing classification},
  author={Liu, Hao and Gao, Yunhao and Li, Wei and Zhang, Mingyang and Gong, Maoguo and Bruzzone, Lorenzo},
  journal={Pattern Recognition},
  volume = {180},
  pages={114309},
  year={2026},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2026.114309},
  publisher={Elsevier}
}
```

## Acknowledgements

Thanks to the following projects for their code and data:

Multimodal Remote Sensing Benchmark Datasets for Land Cover Classification with A Shared and Specific Feature Learning Model: https://github.com/danfenghong/ISPRS_S2FL

More Diverse Means Better: Multimodal Deep Learning Meets Remote-Sensing Imagery Classification: https://github.com/danfenghong/IEEE_TGRS_MDL-RS

Spectral-Spatial Mamba for Hyperspectral Image Classification: https://github.com/mengduanjinghua/Spectral-spatial-Mamba-for-HSIC

FcaNet: Frequency Channel Attention Networks: https://github.com/cfzd/FcaNet


