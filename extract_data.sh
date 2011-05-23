#!/bin/bash

for f in ./data/*zip; do
    dir=${f%.*};
    mkdir -p $dir;
    unzip $f -d $dir;
done;
