#!/bin/bash

MAX_THREADS="16" # <--- Include maximum number of threads (physical cores)
THREADS=(1 16) # <--- Include number of threads to execute PYTHON_FILE


# Define the Python file to run
PYTHON_FILE="benchmark/cbm_construction.py"

# Define lists of values for each variable
ITERATIONS=(50) # default value might take to long, consider lowering it.
WARMUPS=(10)
ALPHAS=(0 1 2 4 8 16 32)

# Extract dataset names automatically
DATASETS=("ca-HepPh" "ca-AstroPh" "Cora" "PubMed" "COLLAB" "coPapersCiteseer" "coPapersDBLP" "ogbn-proteins-raw")

# Temporary file to store results
mkdir -p results

RESULTS_FILE="results/compression_metrics_results.txt"
> $RESULTS_FILE
> comp_temp_results.txt

# Generate all possible combinations
for THREAD in "${THREADS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for ITER in "${ITERATIONS[@]}"; do
      for WARMUP in "${WARMUPS[@]}"; do        
        for ALPHA in "${ALPHAS[@]}"; do 
          ARGS="--dataset $DATASET --iterations $ITER --warmup $WARMUP"
          echo "Running: OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY=\"0-$((MAX_THREADS - 1))\" python $PYTHON_FILE $ARGS --alpha $ALPHA"
        
          # Execute the Python script with the environment variables
          OUTPUT=$(OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY="0-$((MAX_THREADS - 1))" python $PYTHON_FILE $ARGS --alpha $ALPHA)
          
          # Extract performance metrics from the output
          ALPH=$(echo "$OUTPUT" | grep -oP "alpha: \K[\d\.]+")
          COMP=$(echo "$OUTPUT" | grep -oP "Compression Ratio: \K[\d\.]+")
          MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
          STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
          MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
          MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")
          
          # Save the results in a structured format
          echo -e "[nthreads: $THREAD]\t[alpha: $ALPH][$ARGS]:\t\t\tCompression Ratio: ${COMP}x\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> comp_temp_results.txt
        done
      done
    done
  done  
done  



# Print a pretty table of the results
column -t -s $'\t' comp_temp_results.txt > $RESULTS_FILE
rm comp_temp_results.txt