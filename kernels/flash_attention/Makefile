PROJECT = flash_attention

VX_SRCS = kernel.cpp
# VX_SRCS = kernel.gemmini.warpspec.cpp
# VX_SRCS = kernel.gemmini.cpp
VX_INCLUDES = flash_impl.hpp ../sgemm_tcore/sgemm_impl.hpp

OPTS ?= -n16

VX_CFLAGS += -I../sgemm_tcore

include ../common.mk
