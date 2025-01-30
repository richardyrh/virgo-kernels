#!/bin/bash

archs=("volta" "ampere" "hopper")
dims=("256" "512" "1024")

if [ -z "$TOOLDIR" ]; then
    echo "error: \$TOOLDIR not set.  Did you run source ci/toolchain_env.sh?"
    exit 1
fi

switch_binaries() {
    local dim="$1"
    local arch="$2"
    dma=1
    [[ "$arch" == "volta" ]] && dma=0
    echo "dma is $dma"
    if [ "$dma" == "1" ]; then
        layout_a="row.swizzle_fp16"
        layout_b="row"
    else
        layout_a="col.swizzle_fp16"
        layout_b="row.swizzle_fp16"
    fi

    args="args.m$1n$1k$1.bin"
    input_a="input.a.rand01.fp16.m$1n$1k$1.$layout_a.bin"
    input_b="input.b.rand01.fp16.m$1n$1k$1.$layout_b.bin"
    check_exists "$args"
    check_exists "$input_a"
    check_exists "$input_b"

    ln -sf -v "$args" "args.bin"
    ln -sf -v "$input_a" "input.a.bin"
    ln -sf -v "$input_b" "input.b.bin"
}

check_exists() {
    if ! [ -f "$1" ]; then
        echo "error: looked for file $1 that does not exist."
        exit 1
    fi
}

# generate operands
for dim in "${dims[@]}"; do
    echo "generating operands for dim $dim"
    python3 generate_operands.py $dim $dim $dim
    mv -v input.a.col.bin input.a.rand01.fp16.m${dim}n${dim}k${dim}.col.swizzle_fp16.bin
    mv -v input.a.row.bin input.a.rand01.fp16.m${dim}n${dim}k${dim}.row.swizzle_fp16.bin
    mv -v input.b.row.bin input.b.rand01.fp16.m${dim}n${dim}k${dim}.row.bin
    mv -v input.b.row.swizzled.bin input.b.rand01.fp16.m${dim}n${dim}k${dim}.row.swizzle_fp16.bin
done

for arch in "${archs[@]}"; do
    git checkout ae-$arch
    git pull

    # re-compile libvortexrt.a
    pushd ../../lib
    make
    popd

    for dim in "${dims[@]}"; do
        echo "compiling GEMM kernel for $arch with dim $dim"

        switch_binaries $dim $arch

        # touch source file to force re-building, as the Makefile does not track
        # binary changes
        touch kernel.cpp

        make CONFIG=gemm.tcore.$arch.dim$dim
    done
done
