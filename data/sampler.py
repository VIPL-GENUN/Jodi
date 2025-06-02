# This file is modified from https://github.com/NVlabs/Sana

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/NVlabs/VILA/tree/main/llava/wids

import warnings
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class DistributedRangedSampler(Sampler):
    """A sampler that samples in chunks and then shuffles the samples within each chunk.

    This preserves locality of reference while still shuffling the data.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        num_samples: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = None,
    ):
        if drop_last is not None:
            warnings.warn("DistributedChunkedSampler does not support drop_last, thus it will be ignored")
        if not dist.is_initialized():
            warnings.warn(
                "DistributedChunkedSampler is called without distributed initialized; assuming single process"
            )
            num_replicas = 1
            rank = 0
        else:
            num_replicas = num_replicas or dist.get_world_size()
            rank = rank or dist.get_rank()
        assert rank >= 0 and rank < num_replicas
        self.shuffle = shuffle
        num_samples = num_samples or len(dataset)
        self.worker_chunk = num_samples // num_replicas
        self.worker_start = rank * self.worker_chunk
        self.worker_end = min((rank + 1) * self.worker_chunk, num_samples)
        self.ranges = range(self.worker_start, self.worker_end)
        self.epoch = 0
        self.step_start = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.ranges)

    def set_start(self, start):
        self.step_start = start

    def __iter__(self):
        ranges = list(self.ranges[self.step_start:])
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(ranges), generator=g)
            ranges = torch.tensor(ranges)[indices].tolist()
            yield from ranges
        else:
            yield from ranges
        self.epoch += 1
