import os
import re
import json
import numpy as np
from PIL import Image
from random import shuffle

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from data.builder import DATASETS

Image.MAX_IMAGE_PIXELS = None


def get_combinition(n: int):
    return (np.arange(2**n, dtype=np.long)[:, None] >> np.arange(n-1, -1, -1)) & 1


def clean_filename(filename):
    match = re.match(r"^(.*?)(\.jpg|\.png)(?:\.(jpg|png))*$", filename, re.IGNORECASE)
    return match.group(1) + match.group(2) if match else filename


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


@DATASETS.register_module()
class JodiDataset(Dataset):
    """
    The data should be organized as follows:

    ```
        data_dir
        ├── image.jsonl
        ├── image
        │   ├── <dir_1>/<name_1>.jpg
        │   ├── <dir_1>/<name_1>.jpg.caption.json
        │   ├── <dir_1>/<name_1>.jpg.info.json
        │   ├── <dir_2>/<name_2>.jpg
        │   ├── <dir_2>/<name_2>.jpg.caption.json
        │   ├── <dir_2>/<name_2>.jpg.info.json
        │   └── ...
        ├── annotation_<CONDITION_1>.jsonl
        ├── annotation_<CONDITION_1>
        │   ├── <dir_1>/<name_1>.jpg
        │   ├── <dir_2>/<name_2>.jpg
        │   └── ...
        ├── annotation_<CONDITION_2>.jsonl
        ├── annotation_<CONDITION_2>
        │   ├── <dir_1>/<name_1>.jpg.png
        │   ├── <dir_2>/<name_2>.jpg.png
        │   └── ...
        └── ...
    ```

    Each line in jsonl file is `{"image_name": "<dir>/<name>.jpg"}`.
    Each caption.json file is `{"<model_name_1>": "<caption>", "<model_name_2>": "<caption>"}`.

    """

    aspect_ratio = {
        "0.25": [512.0, 2048.0],  # 1:4
        "0.33": [576.0, 1728.0],  # 1:3
        "0.4": [640.0, 1600.0],   # 2:5
        "0.5": [704.0, 1408.0],   # 1:2
        "0.67": [768.0, 1152.0],  # 2:3
        "0.75": [864.0, 1152.0],  # 3:4
        "0.82": [896.0, 1088.0],  # 5:6
        "1.0": [1024.0, 1024.0],  # 1:1
        "1.21": [1088.0, 896.0],  # 6:5
        "1.33": [1152.0, 864.0],  # 4:3
        "1.5": [1152.0, 768.0],   # 3:2
        "2.0": [1408.0, 704.0],   # 2:1
        "2.5": [1600.0, 640.0],   # 5:2
        "3.0": [1728.0, 576.0],   # 3:1
        "4.0": [2048.0, 512.0],   # 4:1
    }

    ratio_nums = {k: 0 for k in aspect_ratio.keys()}

    def __init__(
            self,
            data_dir: str,
            conditions: list[str],
            split: str = None,
            tasks: list[str] = ("j", "c", "p"),
            caption_model_probs: dict[str, float] = None,
            repeat_time: int = 1,
            use_empty_openpose_image: bool = True,
    ):
        self.data_dir = os.path.expanduser(data_dir)
        self.conditions = conditions
        self.split = split
        self.tasks = tasks
        self.caption_model_probs = caption_model_probs
        self.repeat_time = repeat_time
        self.use_empty_openpose_image = use_empty_openpose_image

        self.data = self.load_data()
        self.data = self.data * repeat_time

        self.role_cond_gen = get_combinition(len(self.conditions)) + 1  # (2^K, K)
        self.role_img_perc = get_combinition(len(self.conditions)) * 2  # (2^K, K)

    def load_data(self):
        data = {}
        jsonl_ext = ".jsonl" if self.split is None else f".{self.split}.jsonl"

        if os.path.isdir(os.path.join(self.data_dir, "image_original_size_png")):
            imagedir = os.path.join(self.data_dir, "image_original_size_png")
            imagejson = os.path.join(self.data_dir, "image_original_size_png" + jsonl_ext)
        else:
            imagedir = os.path.join(self.data_dir, "image")
            imagejson = os.path.join(self.data_dir, "image" + jsonl_ext)
        assert os.path.isdir(imagedir), f"Image directory {imagedir} does not exist."
        assert os.path.isfile(imagejson), f"Image json file {imagejson} does not exist."
        with open(imagejson, "r", encoding="utf-8") as f:
            for line in f:
                image_name = json.loads(line)["image_name"]
                image_id = clean_filename(image_name)
                image_path = os.path.join(imagedir, image_name)
                caption_path = image_path + ".caption.json"
                info_path = image_path + ".info.json"
                data[image_id] = {"image_path": image_path, "caption_path": caption_path, "info_path": info_path}
                assert os.path.isfile(image_path), f"Image path {image_path} does not exist."
                assert os.path.isfile(caption_path), f"Caption path {caption_path} does not exist."
                assert os.path.isfile(info_path), f"Info path {info_path} does not exist."

        for condition in self.conditions:
            anndir = os.path.join(self.data_dir, f"annotation_{condition}")
            annjson = os.path.join(self.data_dir, f"annotation_{condition}" + jsonl_ext)
            if os.path.isfile(annjson) and os.path.isdir(anndir):
                with open(annjson, "r", encoding="utf-8") as f:
                    for line in f:
                        image_name = json.loads(line)["image_name"]
                        image_id = clean_filename(image_name)
                        ann_path = os.path.join(anndir, image_name)
                        if image_id in data:
                            data[image_id].update({f"ann_path_{condition}": ann_path})
                            assert os.path.isfile(ann_path), f"Ann path {ann_path} does not exist."

        data = list(data.items())
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_id, info = self.data[index]
        image_path = info["image_path"]

        with open(info["info_path"], "r", encoding="utf-8") as f:
            image_info = json.load(f)
        transform = self.get_transform(image_info["height"], image_info["width"])

        img = [transform(Image.open(image_path).convert("RGB"))]
        role, cnt = [], 0
        for condition in self.conditions:
            ann_path = info.get(f"ann_path_{condition}", None)  # noqa
            if ann_path is None:
                if self.use_empty_openpose_image and "openpose" in condition:
                    h, w = img[0].shape[-2:] if isinstance(img[0], torch.Tensor) else (img[0].height, img[0].width)
                    img.append(transform(self.get_empty_openpose_image(h, w)))
                    role.append(None)
                    cnt += 1
                else:
                    img.append(torch.zeros_like(img[0]) if isinstance(img[0], torch.Tensor) else None)
                    role.append(2)
            else:
                img.append(transform(Image.open(ann_path).convert("RGB")))
                role.append(None)
                cnt += 1

        if isinstance(img[0], torch.Tensor):
            img = torch.stack(img, dim=0)  # (1+K, C, H, W)

        with open(info["caption_path"], "r", encoding="utf-8") as f:
            text_dict = json.load(f)
            if self.caption_model_probs is not None:
                p = list(self.caption_model_probs.values())
                p = np.array(p) / np.sum(p)
                caption_model_id = np.random.choice(len(self.caption_model_probs), p=p)
                caption_model_key = list(self.caption_model_probs.keys())[caption_model_id]
            else:
                caption_model_id = np.random.randint(0, len(text_dict))
                caption_model_key = list(text_dict.keys())[caption_model_id]
            text = text_dict[caption_model_key]

        task = self.tasks[np.random.randint(len(self.tasks))]
        if task == "j":  # joint generation
            role = [0] + [r if r is not None else 0 for r in role]
        elif task == "c":  # condition generation
            _role = self.role_cond_gen[:2**cnt, len(self.conditions)-cnt:]
            _role = _role[np.random.randint(len(_role))].tolist()
            role = [0] + [r if r is not None else _role.pop() for r in role]
        elif task == "p":  # image perception
            _role = self.role_img_perc[:2**cnt, len(self.conditions)-cnt:][:-1]
            _role = _role[np.random.randint(len(_role))].tolist()
            role = [1] + [r if r is not None else _role.pop() for r in role]
        else:
            raise ValueError(f"Unknown task {task}")
        role = torch.tensor(role).long()

        return dict(img=img, text=text, role=role)

    def get_transform(self, height: int, width: int):
        closest_size, closest_ratio = get_closest_ratio(height, width, self.aspect_ratio)
        closest_size = list(map(lambda x: int(x), closest_size))

        if closest_size[0] / height > closest_size[1] / width:
            resize_size = closest_size[0], int(width * closest_size[0] / height)
        else:
            resize_size = int(height * closest_size[1] / width), closest_size[1]

        return T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(closest_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def get_data_info(self, index: int):
        image_id, info = self.data[index]
        with open(info["info_path"], "r", encoding="utf-8") as f:
            image_info = json.load(f)
        return {"height": image_info["height"], "width": image_info["width"]}

    @staticmethod
    def get_empty_openpose_image(h, w):  # pure black image
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


class RandomConcatJodiDataset(Dataset):
    def __init__(self, datasets: list[JodiDataset]):
        self.datasets = datasets
        self.indices = []
        for k, dataset in enumerate(self.datasets):
            self.indices.extend(list(zip([k] * len(dataset), range(len(dataset)))))
        shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        dataset_idx, sample_idx = self.indices[index]
        dataset = self.datasets[dataset_idx]
        return dataset[sample_idx]

    def get_data_info(self, index: int):
        dataset_idx, sample_idx = self.indices[index]
        dataset = self.datasets[dataset_idx]
        return dataset.get_data_info(sample_idx)

    @property
    def aspect_ratio(self):
        return self.datasets[0].aspect_ratio

    @property
    def ratio_nums(self):
        return self.datasets[0].ratio_nums
