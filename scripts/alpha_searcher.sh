#!/bin/bash

MAX_THREADS="16" # <--- Include maximum number of threads (physical cores)
THREADS=(1 16) # <--- Include number of threads to execute PYTHON_FILE

# Define the Python file to run
PYTHON_FILE="benchmark/benchmark_matmul.py"

# Define lists of values for each variable
NCOLUMNS=(500)  # Empty string ensures default value is used when omitted
ITERATIONS=(100)
WARMUPS=(10)
ALPHAS=(0 1 2 4 8 16 32)

# Extract dataset names automatically
DATASETS=("ca-HepPh" "ca-AstroPh" "Cora" "PubMed" "COLLAB" "coPapersCiteseer" "coPapersDBLP" "ogbn-proteins-raw")

# Temporary file to store results
mkdir -p results

RESULTS_FILE="results/alpha_searcher_results.txt"
> $RESULTS_FILE
> alpha_temp_results.txt


# Generate all possible combinations
for THREAD in "${THREADS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for COL in "${NCOLUMNS[@]}"; do
      for ITER in "${ITERATIONS[@]}"; do
        for WARMUP in "${WARMUPS[@]}"; do
          
          ARGS="--dataset $DATASET --iterations $ITER --warmup $WARMUP --columns $COL"
          
          echo "Running: OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY=\"0-$(($MAX_THREADS-1))\" python benchmark/benchmark_matmul.py --operation mkl-ax $ARGS"

          OUTPUT=$(OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY="0-$(($MAX_THREADS - 1))" python benchmark/benchmark_matmul.py --operation mkl-ax $ARGS)
            
          # Extract performance metrics from the output
          MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
          STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
          MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
          MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")

          # Save the results in a structured format
          echo -e "[$THREAD][$ARGS]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> $RESULTS_FILE

          for ALPHA in "${ALPHAS[@]}"; do  
            echo "Running: OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY=\"0-$(($MAX_THREADS-1))\" python benchmark/benchmark_matmul.py --operation cbm-ax $ARGS --alpha $ALPHA"

            # Execute the Python script with the environment variables
            OUTPUT=$(OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY="0-$(($MAX_THREADS - 1))" python benchmark/benchmark_matmul.py --operation cbm-ax $ARGS --alpha $ALPHA)

            # Extract performance metrics from the output
            MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
            STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
            MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
            MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")
            
            # Save the results in a structured format
            echo -e "[$THREAD][$ARGS --alpha $ALPHA]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> alpha_temp_results.txt
          done
        done
      done
    done
  done
done

# Print a pretty table of the results
column -t -s $'\t' alpha_temp_results.txt > $RESULTS_FILE
rm alpha_temp_results.txt