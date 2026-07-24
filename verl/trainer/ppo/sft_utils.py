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

from collections import defaultdict
from typing import Any

import torch


def is_sft_enabled(sft_loss_coeff: float, current_step: int, sft_start_step: int) -> bool:
    """Return whether dynamic SFT is active for the current 1-based global step."""
    return sft_loss_coeff > 0.0 and current_step >= sft_start_step


def build_shortest_correct_sft_mask(
    uids,
    response_mask: torch.Tensor,
    reward_extra_infos: dict[str, Any],
    token_level_scores: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Select the shortest correct rollout for every prompt UID."""
    if len(uids) != response_mask.shape[0]:
        raise ValueError(f"UID count {len(uids)} does not match batch size {response_mask.shape[0]}")

    if "acc" in reward_extra_infos:
        correctness = [bool(value) for value in reward_extra_infos["acc"]]
    else:
        correctness = token_level_scores.sum(dim=-1).gt(0.5).detach().cpu().tolist()

    if len(correctness) != len(uids):
        raise ValueError(f"Correctness count {len(correctness)} does not match UID count {len(uids)}")

    response_lengths = response_mask.sum(dim=-1).detach().cpu().tolist()
    prompt_groups: dict[Any, list[int]] = defaultdict(list)
    for index, uid in enumerate(uids):
        prompt_groups[uid].append(index)

    selected = torch.zeros(len(uids), dtype=torch.bool, device=response_mask.device)
    prompts_with_correct_response = 0
    for indices in prompt_groups.values():
        correct_indices = [index for index in indices if correctness[index]]
        if not correct_indices:
            continue
        shortest_index = min(correct_indices, key=lambda index: response_lengths[index])
        selected[shortest_index] = True
        prompts_with_correct_response += 1

    prompts_total = len(prompt_groups)
    stats = {
        "sft/prompts_with_correct_response": prompts_with_correct_response,
        "sft/prompts_total": prompts_total,
        "sft/correct_response_ratio": prompts_with_correct_response / max(prompts_total, 1),
    }
    return selected, stats
