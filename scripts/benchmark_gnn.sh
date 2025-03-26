#!/bin/bash

# Define the Python file to run
PYTHON_FILE="benchmark/benchmark_gnn.py"

# Define lists of values for each variable
NN_MODELS=("pyg-gcn" "cbm-gcn" "mkl-gcn")
HIDDEN_FEATURES=(10 20)
NUM_HIDDEN_LAYERS=(0)
EPOCHS=(10)
CRITERION=("mse")
OPTIMIZERS=("sgd")
LEARNING_RATES=(0.01)
WARMUPS=(10)
TRAIN_OPTIONS=("" "--train")  # Allows testing both training and inference
FAKE_OPTIONS=("--fake")  # Allows testing with and without fake data

# Define ALPHAS implicitly by dataset (only datasets that appear here are used)
declare -A ALPHA_MAP
ALPHA_MAP["ca-HepPh"]="1 2"  # Space-separated values instead of an array
ALPHA_MAP["ca-AstroPh"]=""  # Empty string means revert to default (no --alpha)

# Extract dataset names automatically
DATASETS=(${!ALPHA_MAP[@]})

# Temporary file to store results
RESULTS_FILE="results.txt"
> $RESULTS_FILE

# Generate all possible combinations
for NN in "${NN_MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for HIDDEN in "${HIDDEN_FEATURES[@]}"; do
      for LAYERS in "${NUM_HIDDEN_LAYERS[@]}"; do
        for EPOCH in "${EPOCHS[@]}"; do
          for CRIT in "${CRITERION[@]}"; do
            for OPT in "${OPTIMIZERS[@]}"; do
              for LR in "${LEARNING_RATES[@]}"; do
                for WARMUP in "${WARMUPS[@]}"; do
                  for TRAIN in "${TRAIN_OPTIONS[@]}"; do
                    for FAKE in "${FAKE_OPTIONS[@]}"; do
                      # Read alpha values as an array
                      IFS=' ' read -r -a ALPHAS <<< "${ALPHA_MAP[$DATASET]}"
                      if [ -z "${ALPHAS[*]}" ]; then
                        ALPHAS=("")  # Ensures at least one iteration without --alpha
                      fi
                      for ALPHA in "${ALPHAS[@]}"; do
                        ARGS="--nn $NN $TRAIN --dataset $DATASET --hidden_features $HIDDEN --num_hidden_layers $LAYERS --epochs $EPOCH --criterion $CRIT --optimizer $OPT --lr $LR --warmup $WARMUP $FAKE"
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
          done
        done
      done
    done
  done
done

# Print a pretty table of the results
column -t -s $'\t' $RESULTS_FILE