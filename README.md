<h1 align="center">
<a href="https://VIPL-GENUN.github.io/Project-Jodi"><img src="assets/logo.png" width="166"></a>
</h1>

<a href="https://arxiv.org/abs/2505.19084"><img src="https://img.shields.io/badge/arXiv-3A98B9?label=%F0%9F%93%9D&labelColor=FFFDD0" style="height: 28px" /></a>
<a href="https://VIPL-GENUN.github.io/Project-Jodi"><img src="https://img.shields.io/badge/Project Page-3A98B9?label=%F0%9F%8F%A0&labelColor=FFFDD0" style="height: 28px" /></a>
<a href="https://huggingface.co/VIPL-GENUN/Jodi"><img src="https://img.shields.io/badge/Model-3A98B9?label=%F0%9F%A4%97&labelColor=FFFDD0" style="height: 28px" /></a>
<a href="https://huggingface.co/datasets/VIPL-GENUN/Joint-1.6M-1024px"><img src="https://img.shields.io/badge/Dataset-3A98B9?label=%F0%9F%A4%97&labelColor=FFFDD0" style="height: 28px" /></a>

> **[Jodi: Unification of Visual Generation and Understanding via Joint Modeling](https://arxiv.org/abs/2505.19084)** \
> [Yifeng Xu](https://xyfjason.github.io/homepage)<sup>1,2</sup>, [Zhenliang He](https://lynnho.github.io)<sup>1</sup>, [Meina Kan](https://scholar.google.com/citations?user=4AKCKKEAAAAJ)<sup>1,2</sup>, [Shiguang Shan](https://scholar.google.com/citations?user=Vkzd7MIAAAAJ)<sup>1,2</sup>, [Xilin Chen](https://scholar.google.com/citations?user=vVx2v20AAAAJ)<sup>1,2</sup> \
> <sup>1</sup>State Key Lab of AI Safety, Institute of Computing Technology, CAS, China \
> <sup>2</sup>University of Chinese Academy of Sciences, China

<p align="center">
    <img src="./assets/banner.jpg" alt="banner" style="width: 100%" />
</p>

We introduce Jodi, a diffusion framework that unifies visual generation and understanding by jointly modeling the image domain and multiple label domains. Jodi is built upon a linear diffusion transformer with a role switch mechanism, enabling joint generation, controllable generation, and image perception in a unified diffusion model.



## 💥 News

- **[2025-06-17]**: The training code and [Joint-1.6M dataset](https://huggingface.co/datasets/VIPL-GENUN/Joint-1.6M-1024px) are released.
- **[2025-05-27]**: The [arXiv paper](https://arxiv.org/abs/2505.19084), [model weights](https://huggingface.co/VIPL-GENUN/Jodi), and inference code are released.



## 🛠️ Installation

> The code is tested with python 3.10.0, torch 2.4.0, and cuda 12.1.

Clone this repo:

```shell
git clone https://github.com/VIPL-GENUN/Jodi.git
cd Jodi
```

Create and activate a new conda environment:

```shell
conda create -n jodi python=3.10.0 -y
conda activate jodi
```

Install dependencies:

```shell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```



## 🤖️ Download Models

We provide our model on [HuggingFace](https://huggingface.co/VIPL-GENUN/Jodi).
The model will be automatically downloaded when you launch the Gradio demo,
or you can download it manually using the following command:

```shell
huggingface-cli download VIPL-GENUN/Jodi
```



## 🚀 Gradio Demo

```shell
python app/jodi_gradio.py --model_path hf://VIPL-GENUN/Jodi/Jodi.pth
```



## 🔥 Training

### Step 1: Data Preparation

We provide the Joint-1.6M dataset on [HuggingFace](https://huggingface.co/datasets/VIPL-GENUN/Joint-1.6M-1024px).
To help you get started quickly, we also provide a small example dataset in `assets/example_data` with the same file structure as the Joint-1.6M dataset.

```text
assets/example_data
├── metadata.jsonl
├── images
│   ├── 0adbfa3cab59b674b83f24a7964ae23f.jpg
│   ├── 0aded2a84831be7b912ef85f6c1eb6e2.jpg
│   └── 0adf204564879c270bafba334ca99e3c.jpg
├── annotation_edge
│   └── (same as images)
├── annotation_depth
│   └── (same as images)
└── ...
```

The code will load the data based on `metadata.jsonl`.
Each line in `metadata.jsonl` is a dictionary containing paths to an image and its annotations (labels), height and width, and captions from different models. For example:

```python
{
  "image": "images/0adbfa3cab59b674b83f24a7964ae23f.jpg",
  "info": {"height": 1280, "width": 1024},
  "caption": {"Qwen2-VL-7b-Instruct": "xxxxxxxx", "BLIP2-OPT-2.7b": "yyy"},
  "annotation_edge": "annotation_edge/0adbfa3cab59b674b83f24a7964ae23f.jpg",
  "annotation_depth": "annotation_depth/0adbfa3cab59b674b83f24a7964ae23f.jpg",
  # ...
}
```

### Step 2: Download Models

Jodi is built on top of [Sana](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16).
You can either finetune Jodi or directly train the model from Sana.

```shell
# download Jodi
huggingface-cli download VIPL-GENUN/Jodi
# download Sana
huggingface-cli download Efficient-Large-Model/Sana_1600M_1024px_BF16
```

### Step 3: Start Training

Finetune Jodi on example dataset:

```shell
bash scripts/train_from_jodi.sh ./configs/train_example_data.yaml
```

Train from Sana on Joint-1.6M dataset:

```shell
bash scripts/train_from_sana.sh ./configs/train_joint1.6m.yaml
```



## 🪧 Acknowledgement

This project is built upon [Sana](https://github.com/NVlabs/Sana). Thanks for their great work!



## ✏️ Citation

If you find this project helpful, please consider citing:

```bibtex
@article{xu2025jodi,
  title={Jodi: Unification of Visual Generation and Understanding via Joint Modeling},
  author={Xu, Yifeng and He, Zhenliang and Kan, Meina and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2505.19084},
  year={2025}
}
```
