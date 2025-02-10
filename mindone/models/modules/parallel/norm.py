from typing import Literal, Optional

from mindspore import dtype as mstype
from mindspore import mint, nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from .param_wrapper import ZeroParamWrapper


class LayerNorm(nn.Cell):
    def __init__(
        self,
        net: mint.nn.LayerNorm,
        zero_stage: Literal[0, 1, 2, 3] = 0,
        op_group: str = GlobalComm.WORLD_COMM_GROUP,
        cell_type: Optional[mstype.Type] = None,
    ):
        super().__init__(auto_prefix=False)
        self.net = net
        self.set_param_wrapper(zero_stage, op_group, cell_type)

    def set_param_wrapper(self, zero_stage, op_group, cell_type=None):
        self.param_wrapper_w = nn.Identity()
        self.param_wrapper_b = nn.Identity()
        if zero_stage == 3:
            # Init parallel settings
            is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
            op_group_size = get_group_size(op_group) if is_parallel else 1
            op_rank_id = get_rank(op_group) if is_parallel else 0
            split_op = ops.Split(0, op_group_size)
            if self.net.weight is not None:
                self.param_wrapper_w = ZeroParamWrapper(self.net.weight, zero_stage, op_group, cell_type)
                if self.param_wrapper_w.need_rewrite:
                    self.net.weight.assign_value(split_op(self.net.weight)[op_rank_id])
            if self.net.bias is not None:
                self.param_wrapper_b = ZeroParamWrapper(self.net.bias, zero_stage, op_group, cell_type)
                if self.param_wrapper_b.need_rewrite:
                    self.net.bias.assign_value(split_op(self.net.bias)[op_rank_id])

    def construct(self, input):
        y = ops.layer_norm(
            input,
            self.net.normalized_shape,
            self.param_wrapper_w(self.net.weight),
            self.param_wrapper_b(self.net.bias),
            self.net.eps,
        )
        return y
