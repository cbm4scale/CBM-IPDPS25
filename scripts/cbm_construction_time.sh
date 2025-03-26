#!/bin/bash

# Define the Python file to run
PYTHON_FILE="benchmark/cbm_construction.py"

# Define lists of values for each variable
NN_MODELS=("ax" "dadx")
ITERATIONS=(50)
WARMUPS=(10)
ALPHAS=(0 1 2 4 8 16 32)

# Define ALPHAS implicitly by dataset (only datasets that appear here are used)
declare -A ALPHA_MAP
ALPHA_MAP["ca-HepPh"]=""  # Space-separated values instead of an array
ALPHA_MAP["ca-AstroPh"]=""  # Empty string means revert to default (no --alpha)

# Extract dataset names automatically
DATASETS=("ca-HepPh" "ca-AstroPh" "Cora" "PubMed" "COLLAB" "coPapersCiteseer" "coPapersDBLP" "ogbn-proteins-raw")

# Temporary file to store results
RESULTS_FILE="construction_time.txt"
> $RESULTS_FILE

# Generate all possible combinations
for NN in "${NN_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for ITER in "${ITERATIONS[@]}"; do
      for WARMUP in "${WARMUPS[@]}"; do
        # Read alpha values as an array
        ARGS="--dataset $DATASET --iterations $ITER --warmup $WARMUP"
        echo "Running: OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY=\"0-15\" python $PYTHON_FILE --nn mkl-$NN $ARGS"
      
        # Execute the Python script with the environment variables
        OUTPUT=$(OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python $PYTHON_FILE --nn mkl-$NN $ARGS)
        
        # Extract performance metrics from the output
        MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
        STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
        MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
        MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")
        
        # Save the results in a structured format
        echo -e "[mkl-$NN][$ARGS]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> $RESULTS_FILE

        for ALPHA in "${ALPHAS[@]}"; do 
          ARGS="--dataset $DATASET --iterations $ITER --warmup $WARMUP"
          echo "Running: OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY=\"0-15\" python $PYTHON_FILE --nn cbm-$NN $ARGS --alpha $ALPHA"
        
          # Execute the Python script with the environment variables
          OUTPUT=$(OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0-15" python $PYTHON_FILE --nn cbm-$NN $ARGS --alpha $ALPHA)
          
          # Extract performance metrics from the output
          MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
          STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
          MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
          MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")
          
          # Save the results in a structured format
          echo -e "[cbm-$NN][$ARGS]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> $RESULTS_FILE
        done
      done
    done
  done
done

# Print a pretty table of the results
column -t -s $'\t' $RESULTS_FILE
