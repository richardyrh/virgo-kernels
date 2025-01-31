#!/bin/bash

# Copyright 2023 blaise
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

ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AE_TOOLCHAIN_DIR="$(realpath ${ENV_SCRIPT_DIR}/../../toolchain)"

export TOOLDIR=${AE_TOOLCHAIN_DIR}/vortex-toolchain-prebuilt

export VERILATOR_ROOT=$TOOLDIR/verilator
export PATH=$VERILATOR_ROOT/bin:$PATH

export SV2V_PATH=$TOOLDIR/sv2v
export PATH=$SV2V_PATH/bin:$PATH

export YOSYS_PATH=$TOOLDIR/yosys
export PATH=$YOSYS_PATH/bin:$PATH

# LLVM_POCL seems to be only used in tests/opencl
export LLVM_POCL=${AE_TOOLCHAIN_DIR}/llvm-vortex2
export LLVM_VORTEX=${AE_TOOLCHAIN_DIR}/llvm-vortex2
export POCL_CC_PATH=${AE_TOOLCHAIN_DIR}/pocl-vortex2/compiler
export POCL_RT_PATH=${AE_TOOLCHAIN_DIR}/pocl-vortex2/runtime
