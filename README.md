# cbm-benchmark
Refactor of latest version of CBM4Scale. New lean and informative benchmark.

## Setup
The docker should be installed on the local machine as mentioned [here](https://docs.docker.com/get-started/get-docker/).
```bash
# On the local machine
git clone https://github.com/cbm4scale/CBM-IPDPS25.git --recursive
cd CBM-IPDPS25/
docker build -t cbm4gnn .
docker run --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn /bin/bash
# Inside the Docker instance
cd CBM-IPDPS25/
python setup.py
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
