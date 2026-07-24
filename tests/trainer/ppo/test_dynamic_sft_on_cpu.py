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

import os

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict

from verl.trainer.ppo.sft_utils import build_shortest_correct_sft_mask, is_sft_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.utils.losses import compute_shortest_correct_sft_loss, ppo_loss


def test_build_shortest_correct_sft_mask_selects_one_trajectory_per_prompt():
    uids = ["prompt-a", "prompt-a", "prompt-a", "prompt-b", "prompt-b"]
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)

    selected, stats = build_shortest_correct_sft_mask(
        uids=uids,
        response_mask=response_mask,
        reward_extra_infos={"acc": [1, 1, 0, 0, 0]},
        token_level_scores=token_level_scores,
    )

    assert selected.tolist() == [False, True, False, False, False]
    assert stats == {
        "sft/prompts_with_correct_response": 1,
        "sft/prompts_total": 2,
        "sft/correct_response_ratio": 0.5,
    }


def test_build_shortest_correct_sft_mask_falls_back_to_positive_sequence_score():
    uids = ["prompt-a", "prompt-a", "prompt-b", "prompt-b"]
    response_mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    token_level_scores = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [0.2, 0.2, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    selected, stats = build_shortest_correct_sft_mask(
        uids=uids,
        response_mask=response_mask,
        reward_extra_infos={},
        token_level_scores=token_level_scores,
    )

    assert selected.tolist() == [False, True, False, True]
    assert stats["sft/prompts_with_correct_response"] == 2
    assert stats["sft/correct_response_ratio"] == 1.0


@pytest.mark.parametrize(
    ("coefficient", "current_step", "start_step", "expected"),
    [
        (0.0, 10, 1, False),
        (0.01, 14, 15, False),
        (0.01, 15, 15, True),
        (0.01, 16, 15, True),
    ],
)
def test_is_sft_enabled_uses_one_based_global_step(coefficient, current_step, start_step, expected):
    assert is_sft_enabled(coefficient, current_step=current_step, sft_start_step=start_step) is expected


def test_actor_yaml_accepts_dynamic_sft_settings():
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "ppo_micro_batch_size_per_gpu=4",
                "sft_loss_coeff=0.01",
                "sft_start_step=15",
            ],
        )

    config = omega_conf_to_dataclass(cfg)
    assert config.sft_loss_coeff == pytest.approx(0.01)
    assert config.sft_start_step == 15


def test_ppo_loss_adds_weighted_sft_loss_to_pg_loss():
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor"), version_base=None):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "ppo_micro_batch_size_per_gpu=4",
                "rollout_n=2",
                "sft_loss_coeff=0.01",
                "sft_start_step=15",
            ],
        )
    config = omega_conf_to_dataclass(cfg)

    batch_size = 4
    prompts = torch.nested.as_nested_tensor([torch.tensor([1])] * batch_size, layout=torch.jagged)
    responses = torch.nested.as_nested_tensor([torch.tensor([2, 3])] * batch_size, layout=torch.jagged)
    data = TensorDict(
        {
            "prompts": prompts,
            "responses": responses,
            "response_mask": torch.ones(batch_size, 2, dtype=torch.bool),
            "old_log_probs": torch.full((batch_size, 2), -2.0),
            "advantages": torch.zeros(batch_size, 2),
            "sft_loss_mask": torch.tensor([True, False, True, False]),
        },
        batch_size=[batch_size],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=batch_size)
    model_output = {
        "log_probs": torch.nested.as_nested_tensor(
            [torch.full((3,), -2.0)] * batch_size,
            layout=torch.jagged,
        )
    }

    policy_loss, metrics = ppo_loss(config=config, model_output=model_output, data=data)
    pg_loss = metrics["actor/pg_loss"].values[0]
    sft_loss = metrics["actor/sft_loss"].values[0]
    weighted_sft_loss = metrics["actor/sft_loss_weighted"].values[0]

    assert pg_loss == pytest.approx(0.0)
    assert sft_loss == pytest.approx(2.0)
    assert weighted_sft_loss == pytest.approx(config.sft_loss_coeff * sft_loss)
    assert policy_loss.item() == pytest.approx(pg_loss + weighted_sft_loss)


def test_shortest_correct_sft_loss_is_mean_target_nll_per_prompt():
    log_prob = torch.tensor(
        [
            [-1.0, -3.0, -9.0],
            [-8.0, -8.0, -8.0],
            [-2.0, -4.0, -6.0],
            [-7.0, -7.0, -7.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=torch.bool,
    )
    selected = torch.tensor([True, False, True, False])

    loss = compute_shortest_correct_sft_loss(
        log_prob=log_prob,
        response_mask=response_mask,
        sft_loss_mask=selected,
        dp_size=1,
        global_batch_size=4,
        rollout_n=2,
    )

    # Prompt 1 target NLL is (1 + 3) / 2; prompt 2 target NLL is (2 + 4 + 6) / 3.
    assert loss.item() == pytest.approx(3.0)


@pytest.mark.parametrize("num_micro_batches", [1, 2, 4])
def test_shortest_correct_sft_loss_is_microbatch_invariant(num_micro_batches):
    log_prob = -torch.arange(1, 33, dtype=torch.float32).reshape(8, 4)
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    selected = torch.tensor([False, True, False, False, False, False, True, False])
    kwargs = {"dp_size": 1, "global_batch_size": 8, "rollout_n": 4}

    whole = compute_shortest_correct_sft_loss(log_prob, response_mask, selected, **kwargs)
    step = log_prob.shape[0] // num_micro_batches
    accumulated = sum(
        compute_shortest_correct_sft_loss(
            log_prob[i : i + step],
            response_mask[i : i + step],
            selected[i : i + step],
            **kwargs,
        )
        for i in range(0, log_prob.shape[0], step)
    )

    torch.testing.assert_close(accumulated, whole)


def test_shortest_correct_sft_loss_is_differentiable_when_no_prompt_is_correct():
    log_prob = torch.randn(4, 3, requires_grad=True)
    response_mask = torch.ones(4, 3, dtype=torch.bool)
    selected = torch.zeros(4, dtype=torch.bool)

    loss = compute_shortest_correct_sft_loss(
        log_prob,
        response_mask,
        selected,
        dp_size=1,
        global_batch_size=4,
        rollout_n=2,
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(log_prob.grad, torch.zeros_like(log_prob))
