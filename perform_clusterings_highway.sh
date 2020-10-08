#!/bin/bash
for i in $(seq 2 1 7)
do

  python3 run_clustering_highway.py --num_clusters $i &

done