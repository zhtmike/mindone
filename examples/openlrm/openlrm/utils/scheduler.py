# Copyright (c) 2023-2024, Zexin He
#
# This code is adapted from https://github.com/3DTopia/OpenLRM
# with modifications to run openlrm on mindspore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from logging import getLogger

from mindspore.experimental.optim.lr_scheduler import LRScheduler

logger = getLogger(__name__)


class CosineWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, max_iters: int, initial_lr: float = 1e-10, last_iter: int = -1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.initial_lr = initial_lr
        super().__init__(optimizer, last_iter)

    def get_lr(self):
        logger.debug(f"last epoch: {self.last_epoch} | warmup iters: {self.warmup_iters} | max iters: {self.max_iters}")
        if self.last_epoch <= self.warmup_iters:
            return [
                self.initial_lr + (base_lr - self.initial_lr) * self.last_epoch / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            cos_iter = self.last_epoch - self.warmup_iters
            cos_max_iter = self.max_iters - self.warmup_iters
            cos_theta = cos_iter / cos_max_iter * math.pi
            cos_lr = [base_lr * (1 + math.cos(cos_theta)) / 2 for base_lr in self.base_lrs]
            return cos_lr
