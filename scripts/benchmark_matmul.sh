#!/bin/bash

# Define the Python file to run
PYTHON_FILE="benchmark/benchmark_matmul.py"

# Define lists of values for each variable
NN_MODELS=("cbm-ax" "mkl-ax")
COLUMNS=("" 10)  # Empty string ensures default value is used when omitted
ITERATIONS=(2)
WARMUPS=(1)

# Define ALPHAS implicitly by dataset (only datasets that appear here are used)
declare -A ALPHA_MAP
ALPHA_MAP["ca-HepPh"]=""  # Space-separated values instead of an array
ALPHA_MAP["ca-AstroPh"]=""  # Empty string means revert to default (no --alpha)

# Extract dataset names automatically
DATASETS=(${!ALPHA_MAP[@]})

# Temporary file to store results
RESULTS_FILE="results.txt"
> $RESULTS_FILE

# Generate all possible combinations
for NN in "${NN_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for COL in "${COLUMNS[@]}"; do
      for ITER in "${ITERATIONS[@]}"; do
        for WARMUP in "${WARMUPS[@]}"; do
          # Read alpha values as an array
          IFS=' ' read -r -a ALPHAS <<< "${ALPHA_MAP[$DATASET]}"
          if [ -z "${ALPHAS[*]}" ]; then
            ALPHAS=("")  # Ensures at least one iteration without --alpha
          fi
          for ALPHA in "${ALPHAS[@]}"; do
            ARGS="--nn $NN --dataset $DATASET --iterations $ITER --warmup $WARMUP"
            if [ -n "$COL" ]; then
              ARGS="$ARGS --columns $COL"
            fi
            if [ -n "$ALPHA" ]; then
              ARGS="$ARGS --alpha $ALPHA"
            fi
            
            echo "Running: OMP_NUM_THREADS=24 GOMP_CPU_AFFINITY=\"0-23\" python $PYTHON_FILE $ARGS"
            
            # Execute the Python script with the environment variables
            OUTPUT=$(OMP_NUM_THREADS=24 GOMP_CPU_AFFINITY="0-23" python $PYTHON_FILE $ARGS)
            
            # Extract performance metrics from the output
            MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
            STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
            MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
            MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")
            
            # Save the results in a structured format
            echo -e "[$ARGS]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> $RESULTS_FILE
          done
        done
      done
    done
  done
done

# Print a pretty table of the results
column -t -s $'\t' $RESULTS_FILE
