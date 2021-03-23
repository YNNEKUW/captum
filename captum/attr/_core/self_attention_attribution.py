#!/usr/bin/env python3
import typing
from typing import Any, Callable, List, Tuple, Union

import torch
from torch import Tensor

from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_input,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import (
    BaselineType,
    Literal,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from captum.log import log_usage


class SelfAttentionAttribution(GradientAttribution):
    r"""
    Some descriptions
    """

    def __init__(
        self,
        forward_func: Callable,
        m: float = 20.0,
    ) -> None:
        r"""
        Args:
            forward_func (callable): The forward function of the model or any
                    modificationof it
            m (float): The hyperparameter forRiemman approximation of the integration 
            (https://arxiv.org/abs/1703.01365).
        """
        GradientAttribution.__init__(self, forward_func)
        self.
        self.m = m

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:
            inputs (tensor or tuple of tensors): Input for  which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, i.e., batch size, and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            target (int, tuple, tensor or list, optional):

        """
        is_inputs_tuple = _is_tuple(inputs)

        inputs = _format_input(inputs)

