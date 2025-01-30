#!/bin/sh

# hopper and virgo has the same SIMT configurations
git checkout ae-hopper
git pull

# re-compile libvortexrt.a
pushd ../../lib
make
popd

if [ ! -f input.a.rand01.fp16.m256n256k256.row.bin ]; then
    echo "input binaries not found, generating operands"
    python3 generate_operands.py
fi

for a in args/*; do
    echo "compiling GEMM kernel for Virgo with dim ${a}"
    cp -f $a args.bin
    aa=$(basename "$a")
    cp -f input.a.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.a.bin
    cp -f input.b.rand01.fp16.m${aa}n${aa}k${aa}.row.bin input.b.bin
    touch input.c.bin

    # touch source file to force re-building, as the Makefile does not track
    # binary changes
    touch kernel.cpp

    make CONFIG=gemm.virgo.hopper.dim${aa}
done
