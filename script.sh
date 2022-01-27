#!/bin/bash
array=("CTHREADS" "OMP")
#echo "" > results.txt
for i in ${array[@]}; do
  for ((j = 2; j <= 16; j=j+2)); do
    echo "${i} ${j}" >> results.txt
    demo/demo --timing-mode --implementation $i --threads $j hugeScenario.xml | grep "Speedup" | awk '{print $2}' >> results.txt
  done
done

