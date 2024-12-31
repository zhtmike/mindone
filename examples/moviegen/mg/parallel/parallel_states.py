import logging
from typing import List, Optional

import numpy as np

from mindspore.communication import create_group, get_group_size, get_rank

__all__ = [
    "set_data_parallel_group",
    "get_data_parallel_group",
    "set_tensor_parallel_group",
    "get_tensor_parallel_group",
    "set_context_parallel_group",
    "get_context_parallel_group",
    "create_parallel_group",
]

_logger = logging.getLogger()


_GLOBAL_PARALLEL_GROUPS = dict()


def set_data_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["data"] = group


def get_data_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("data", None)


def set_tensor_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["tensor"] = group


def get_tensor_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("tensor", None)


def set_context_parallel_group(group: str) -> None:
    _GLOBAL_PARALLEL_GROUPS["context"] = group


def get_context_parallel_group() -> Optional[str]:
    return _GLOBAL_PARALLEL_GROUPS.get("context", None)


def create_parallel_group(tensor_parallel_shards: int = 1, context_parallel_shards: int = 1) -> None:
    device_num = get_group_size()
    if device_num % tensor_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be divisible by the number of tensor parallel shards ({tensor_parallel_shards})."
        )

    if device_num % context_parallel_shards != 0:
        raise ValueError(
            f"Total number of devices ({device_num}) must be divisible by the number of context parallel shards ({context_parallel_shards})."
        )

    data_parallel_shards = device_num // tensor_parallel_shards // context_parallel_shards
    if data_parallel_shards < 1:
        raise ValueError(
            f"tensor parallel shards ({tensor_parallel_shards}) and context parallel shards ({context_parallel_shards}) "
            f"must be less than the total number of devices ({device_num})."
        )

    # create id mesh
    rank_ids = np.arange(device_num).reshape((data_parallel_shards, tensor_parallel_shards, context_parallel_shards))
    dp_rank_id_pairs = rank_ids.transpose(1, 2, 0).reshape(-1, data_parallel_shards)
    tp_rank_id_pairs = rank_ids.transpose(0, 2, 1).reshape(-1, tensor_parallel_shards)
    cp_rank_id_pairs = rank_ids.reshape(-1, context_parallel_shards)

    # identiy which group the current group belongs to
    my_rank_id = get_rank()
    my_dp_group_id = np.where(my_rank_id == dp_rank_id_pairs)[0].squeeze().item()
    my_tp_group_id = np.where(my_rank_id == tp_rank_id_pairs)[0].squeeze().item()
    my_cp_group_id = np.where(my_rank_id == cp_rank_id_pairs)[0].squeeze().item()

    my_dp_group_name = f"dp_group_{my_dp_group_id}"
    _create_group(my_dp_group_name, dp_rank_id_pairs[my_dp_group_id].tolist())
    set_data_parallel_group(my_dp_group_name)

    my_tp_group_name = f"tp_group_{my_tp_group_id}"
    _create_group(my_tp_group_name, tp_rank_id_pairs[my_tp_group_id].tolist())
    set_tensor_parallel_group(my_dp_group_name)

    my_cp_group_name = f"cp_group_{my_cp_group_id}"
    _create_group(my_cp_group_name, cp_rank_id_pairs[my_cp_group_id].tolist())
    set_context_parallel_group(my_cp_group_name)


def _create_group(group: str, rank_ids: List[int]) -> None:
    _logger.info(f"create group `{group}` with rank ids {rank_ids}.")
    return create_group(group, rank_ids)
