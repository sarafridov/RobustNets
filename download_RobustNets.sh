#!/bin/bash

## Declare list of pruning methods
declare -a pruning_methods=("FT" "GMP" "lth" "lrr" "biprop" "edgepopup")

## Download and extract release files from RobustNets github
cd RobustNets
for p in "${pruning_methods[@]}"
do
    ## Download model state files for current pruning method
    wget https://github.com/sarafridov/RobustNets/releases/download/v1.0.0-robustnets/${p}.tar.gz
    tar -xvzf ${p}.tar.gz -C .
    rm ${p}.tar.gz
    mv ${p}/* .
    rmdir ${p}
done
