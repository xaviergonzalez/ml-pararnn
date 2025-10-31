#
#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import typing as typ
import typing_inspect
from functools import partial
import abc

import torch
import math

from pararnn.rnn_cell.rnn_cell_utils import SystemParametersT, ConfigT
from pararnn.rnn_cell.rnn_cell_application import RNNCellApplicationMode
from pararnn.rnn_cell.rnn_cell_application import RNNCellSequentialApplication, \
    RNNCellParallelApplication, RNNCellParallelCUDAApplication, RNNCellParallelFusedApplication

from pararnn.rnn_cell.rnn_cell_impl import RNNCellImplT
from pararnn.parallel_reduction.parallel_reduction import NewtonConfig

from pararnn.utils.nonlinearities import NONLIN_REGISTRY, NONLIN_AND_DERIVATIVE_REGISTRY


class BaseRNNCell(torch.nn.Module, abc.ABC, typ.Generic[ConfigT, SystemParametersT, RNNCellImplT]):
    
    def __init_subclass__(cls, **_: typ.Any) -> None:
        super().__init_subclass__()
        # Check templates traits match
        generic_base_type, = typing_inspect.get_generic_bases(cls)
        config_type, system_parameters_type, recurrent_model_impl_type = typing_inspect.get_args(generic_base_type)
        assert config_type.trait() is system_parameters_type.trait()
        assert system_parameters_type.trait() is recurrent_model_impl_type.trait()

    @classmethod
    @typ.final
    def _get_impl_type(cls) -> typ.Type[RNNCellImplT]:
        generic_base_type, = typing_inspect.get_generic_bases(cls)
        _, _, recurrent_model_impl_type = typing_inspect.get_args(generic_base_type)
        return recurrent_model_impl_type

    @classmethod
    @typ.final
    def _get_system_parameters_type(cls) -> typ.Type[SystemParametersT]:
        generic_base_type, = typing_inspect.get_generic_bases(cls)
        _, system_parameters_type, _ = typing_inspect.get_args(generic_base_type)
        return system_parameters_type
    
    @property
    def system_parameters(self) -> SystemParametersT:
        return self._system_parameters
    
    @abc.abstractmethod
    def _specific_init(self, config: ConfigT):
        ...
    
    def __init__(self, config: ConfigT):
        super().__init__()
        self.impl_type = self._get_impl_type()
        self.system_parameters_type = self._get_system_parameters_type()

        assert config.mode in [mode.value for mode in RNNCellApplicationMode], \
            f"Selected model application mode '{config.mode}' not recognised"
        self._config = config
        if isinstance(config.newton_config, NewtonConfig):
            self.newton_config = config.newton_config
        else:
            self.newton_config = NewtonConfig(**config.newton_config)
        
        self.device = config.device
        self.dtype = config.dtype
        
        self.state_dim = config.state_dim
        self.input_dim = config.input_dim
        
        # self._system_parameters = None
        self._mode = None
        self._fw_fn = None
        
        self._specific_init(config)
        
        self.__post_init__()
    
    def __post_init__(self):
        self._mode = self.mode = RNNCellApplicationMode(self._config.mode)

    def forward(self, x):
        return self._fw_fn(x)

    
    @classmethod
    def _set_nonlinearity(cls, nonlin: str) -> typ.Callable:
        return NONLIN_REGISTRY[nonlin]

    @classmethod
    def _set_nonlinearity_and_derivative(cls, nonlin: str) -> typ.List[typ.Callable]:
        return NONLIN_AND_DERIVATIVE_REGISTRY[nonlin]
    
    @property
    def mode(self) -> RNNCellApplicationMode:
        return self._mode
    
    @mode.setter
    def mode(self, mode: RNNCellApplicationMode) -> None:
        
        if mode is RNNCellApplicationMode.SEQUENTIAL:
            self._fw_fn = partial(
                RNNCellSequentialApplication.forward,
                state_dim=self.state_dim,
                impl=self.impl_type,
                system_parameters=self._system_parameters,
            )
        elif mode is RNNCellApplicationMode.PARALLEL:
            def fw_fn(x):
                return RNNCellParallelApplication.apply(
                    x,
                    self.state_dim,
                    self.impl_type,
                    self.system_parameters_type,
                    self.newton_config,
                    *self._system_parameters.unpack()  # must unpack because forward doesn't like containers
                )
            self._fw_fn = fw_fn
        elif mode is RNNCellApplicationMode.PARALLEL_CUDA:
            def fw_fn(x):
                return RNNCellParallelCUDAApplication.apply(
                    x,
                    self.state_dim,
                    self.impl_type,
                    self.system_parameters_type,
                    self.newton_config,
                    *self._system_parameters.unpack()
                )
            self._fw_fn = fw_fn         
        elif mode is RNNCellApplicationMode.PARALLEL_FUSED:
            def fw_fn(x):
                return RNNCellParallelFusedApplication.apply(
                    x,
                    self.state_dim,
                    self.impl_type,
                    self.system_parameters_type,
                    self.newton_config,
                    *self._system_parameters.unpack()
                )
            self._fw_fn = fw_fn
        else:
            raise ValueError(f"Unrecognized option '{mode}'.")
        
        self._fw_fn = self._fw_fn
        self._mode = mode
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        return
    
