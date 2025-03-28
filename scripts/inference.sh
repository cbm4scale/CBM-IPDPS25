#!/bin/bash

MAX_THREADS="16" # <--- Include maximum number of threads (physical cores)
THREADS=(1 16) # <--- Include number of threads to execute PYTHON_FILE

# Define the Python file to run
PYTHON_FILE="benchmark/benchmark_inference.py"

# Define lists of values for each variable
NN_MODELS=("cbm-gcn-inference" "mkl-gcn-inference")
HIDDEN_FEATURES=(500)
NUM_HIDDEN_LAYERS=(1)
EPOCHS=(250)
WARMUPS=(10)
# Define ALPHAS implicitly by dataset (only datasets that appear here are used)
declare -A ALPHA_MAP
ALPHA_MAP["ca-HepPh"]=""  # Space-separated values instead of an array
ALPHA_MAP["ca-AstroPh"]=""  # Empty string means revert to default (no --alpha)
ALPHA_MAP["Cora"]=""  # Space-separated values instead of an array
ALPHA_MAP["PubMed"]=""  # Empty string means revert to default (no --alpha)
ALPHA_MAP["COLLAB"]=""  # Space-separated values instead of an array
ALPHA_MAP["coPapersCiteseer"]=""  # Empty string means revert to default (no --alpha)
ALPHA_MAP["coPapersDBLP"]=""  # Space-separated values instead of an array
ALPHA_MAP["ogbn-proteins-raw"]=""  # Empty string means revert to default (no --alpha)

# Extract dataset names automatically
DATASETS=(${!ALPHA_MAP[@]})

# Temporary file to store results
mkdir -p results

RESULTS_FILE="results/inference_results.txt"
> $RESULTS_FILE
> inf_temp_results.txt

# Generate all possible combinations
for THREAD in "${THREADS[@]}"; do
  for NN in "${NN_MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
      for HIDDEN in "${HIDDEN_FEATURES[@]}"; do
        for LAYERS in "${NUM_HIDDEN_LAYERS[@]}"; do
          for EPOCH in "${EPOCHS[@]}"; do
            for WARMUP in "${WARMUPS[@]}"; do
              # Read alpha values as an array
              IFS=' ' read -r -a ALPHAS <<< "${ALPHA_MAP[$DATASET]}"
              if [ -z "${ALPHAS[*]}" ]; then
                ALPHAS=("")  # Ensures at least one iteration without --alpha
              fi
              for ALPHA in "${ALPHAS[@]}"; do
                ARGS="--nn $NN --dataset $DATASET --hidden_features $HIDDEN --num_hidden_layers $LAYERS --epochs $EPOCH --warmup $WARMUP"
                if [ -n "$ALPHA" ]; then
                  ARGS="$ARGS --alpha $ALPHA"
                fi

                echo "Running: OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY=\"0-$((MAX_THREADS - 1))\" python $PYTHON_FILE $ARGS"

                # Execute the Python script with the environment variables
                OUTPUT=$(OMP_NUM_THREADS=$THREAD GOMP_CPU_AFFINITY="0-$((MAX_THREADS - 1))" python $PYTHON_FILE $ARGS)

                # Extract performance metrics from the output
                MEAN=$(echo "$OUTPUT" | grep -oP "Mean: \K[\d\.]+")
                STD=$(echo "$OUTPUT" | grep -oP "Std: \K[\d\.]+")
                MIN=$(echo "$OUTPUT" | grep -oP "Min: \K[\d\.]+")
                MAX=$(echo "$OUTPUT" | grep -oP "Max: \K[\d\.]+")

                # Save the results in a structured format
                echo -e "[nthreads: $THREAD]\t[$ARGS]:\t\tMean: $MEAN\tStd: $STD\tMin: $MIN\tMax: $MAX" >> inf_temp_results.txt
              done
            done
          done
        done
      done
    done
  done
done

# Print a pretty table of the results
column -t -s $'\t' inf_temp_results.txt > $RESULTS_FILE
rm inf_temp_results.txt