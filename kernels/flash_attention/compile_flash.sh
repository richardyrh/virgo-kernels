#!/bin/bash

archs=("ampere" "virgo")

if [ -z "$TOOLDIR" ]; then
    echo "error: \$TOOLDIR not set.  Did you run source ci/toolchain_env.sh?"
    exit 1
fi

check_exists() {
    if ! [ -f "$1" ]; then
        echo "error: looked for file $1 that does not exist."
        exit 1
    fi
}

# generate operands
echo "generating flash_attn operands for seqlen 1024, headdim 64"
python3 flash_attn.py 1024 64 64
mv -v input.a.col.bin input.a.rand.fp32.seqlen1024headdim64.col.bin
mv -v input.a.row.bin input.a.rand.fp32.seqlen1024headdim64.row.bin
mv -v input.b.bin input.b.rand.fp32.seqlen1024headdim64.row.bin
mv -v input.c.bin input.c.rand.fp32.seqlen1024headdim64.row.bin
ln -sf input.a.rand.fp32.seqlen1024headdim64.row.bin input.a.bin
ln -sf input.b.rand.fp32.seqlen1024headdim64.row.bin input.b.bin
ln -sf input.c.rand.fp32.seqlen1024headdim64.row.bin input.c.bin

for arch in "${archs[@]}"; do
    git checkout ae-flash-$arch
    git pull

    # re-compile libvortexrt.a
    pushd ../../lib
    make
    popd

    echo "compiling flash_attn kernel for $arch with seqlen 1024, headdim 64"

    # touch source file to force re-building, as the Makefile does not track
    # binary changes
    touch kernel.cpp
    touch kernel.gemmini.cpp

    make CONFIG=flash.$arch.seqlen1024.headdim64
done
