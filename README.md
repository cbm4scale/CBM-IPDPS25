# cbm-benchmark
Refactor of latest version of CBM4Scale. New lean and informative benchmark.

## Setup
```bash
git clone git@github.com:cbm4scale/cbm-benchmark.git --recursive            # Clone the repository via SSH
cd cbm-benchmark
git submodule init
git submodule update
python setup.py --setvars_path /home/guests2/tfa/intel/oneapi/setvars.sh    # Set the path to the setvars.sh file
export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
export PYTHONPATH=./:$PYTHONPATH
```

## Grid search
### benchmark_gnn.sh (example)
```bash
NN_MODELS=("pyg-gcn" "cbm-gcn" "mkl-gcn")   # Not comma separated
HIDDEN_FEATURES=(10 20)
NUM_HIDDEN_LAYERS=(0 1 2)
EPOCHS=(10 100)
CRITERION=("mse")                           # Use only one criterion
OPTIMIZERS=("sgd" "adam")
LEARNING_RATES=(0.01)
WARMUPS=(10)
TRAIN_OPTIONS=("" "--train")                # Inference (empty string) and Train (--train)
FAKE_OPTIONS=("" "--fake")
# [...]
ALPHA_MAP["ca-HepPh"]="1 2 4"               # Datasets to use are implicitly defined by the keys of ALPHA_MAP
ALPHA_MAP["ca-AstroPh"]=""                  # Use the default alpha for the respective dataset
```