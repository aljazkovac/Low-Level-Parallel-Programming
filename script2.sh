#!/bin/bash
implementations=("SEQ" "CTHREADS --threads 6" "OMP --threads 8" "SIMD")
scenarios=("scenario.xml" "hugeScenario.xml" "scenario_box.xml")

for implementation in "${implementations[@]}"; do
    echo "${implementation}" >> results2.txt
    for scenario in ${scenarios[@]}; do
        echo "${scenario}" >> results2.txt
        for ((j = 0; j < 2; j=j+1)); do
            demo/demo --timing-mode --implementation $implementation $scenario | grep "Target time" | awk '{print $3}' >> results2.txt
        done
    done
done

