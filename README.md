# Diffusion²: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models
[Paper](https://arxiv.org/abs/2404.02148)
> [**Diffusion²: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models**](https://arxiv.org/abs/2404.02148),            
> Zeyu Yang*, Zijie Pan*, [Chun Gu](https://sulvxiangxin.github.io), [Li Zhang](https://lzrobots.github.io)  
> **Fudan University**  
> **Arxiv preprint**

**This repository is the official implementation of "Diffusion²: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models".** In this paper, we propose to achieve 4D generation from directly sampling the dense multi-view multi-frame observation of dynamic content by composing the estimated score of pretrained video and multi-view diffusion models that have learned strong prior of dynamic and geometry. 

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## 📊 Results

### 🖼️ Image-to-4D Generation

<div align="center">
  <img src="assets/image-to-4D.png"/>
</div><br/>

### 🖼️ 4D Generatin from Single-view Video 

<div align="center">
  <img src="assets/video-to-4D.png"/>
</div><br/>

### 🖼️ Animating Static 3D assets

<div align="center">
  <img src="assets/static-to-4D.png"/>
</div><br/>

### 🖼️ Ablation

<div align="center">
  <img src="assets/ablation.png"/>
</div><br/>

## 📜 Reference
```bibtex
@article{yang2024diffusion,
  title={Diffusion²: Dynamic 3D Content Generation via Score Composition of Orthogonal Diffusion Models},
  author={Yang, Zeyu and Pan, Zijie and Gu, Chun and Zhang, Li},
  journal={arXiv preprint 2404.02148},
  year={2024}
}
```
