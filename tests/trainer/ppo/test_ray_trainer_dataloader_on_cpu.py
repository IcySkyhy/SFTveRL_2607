# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler, default_collate

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_v0_dataloader_uses_train_batch_size_when_gen_batch_size_is_null():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {
                "gen_batch_size": None,
                "train_batch_size": 2,
                "val_batch_size": 2,
                "dataloader_num_workers": 0,
                "validation_shuffle": False,
            },
            "trainer": {"total_epochs": 1, "total_training_steps": None},
        }
    )
    train_dataset = list(range(6))
    val_dataset = list(range(2))

    trainer._create_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=default_collate,
        train_sampler=SequentialSampler(train_dataset),
    )

    assert trainer.train_dataloader.batch_size == trainer.config.data.train_batch_size
    assert len(trainer.train_dataloader) == 3
