# CBM-Benchmark

This repository contains the refactored version of the latest CBM4Scale, featuring a streamlined and informative benchmark for matrix multiplication.

---

## Setup

### Installation with Conda

1. **Install Intel oneAPI Base Toolkit**  
   Download and install the Intel oneAPI Base Toolkit following the instructions provided [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/overview.html).

2. **Create a Conda Environment**  
   Set up a new Conda environment and install the necessary dependencies:
   ```bash
   conda create -n cbm python=3.11
   conda activate cbm
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip uninstall numpy
   pip install numpy==1.24.3
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
   pip install ogb
   conda install cmake ninja wget prettytable scipy
    ```
4. **Clone and Install the Repository**  
   Clone the repository and set up the project:
   ```bash
   git clone https://github.com/cbm4scale/CBM-IPDPS25.git --recursive
   cd CBM-IPDPS25/
   git submodule init
   git submodule update
   python setup.py  # If Intel oneAPI is not installed in your default directory, use: --setvars_path PATH_TO_ONEAPI/setvars.sh
   export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
   export PYTHONPATH=./:$PYTHONPATH
   ```

### Installation with Docker

1. **Install Docker**  
   Download and install Docker following the instructions provided [here](https://docs.docker.com/get-started/get-docker/).

2. **Clone the Repository**  
   Clone the repository and build the Docker image:
   ```bash
   git clone https://github.com/cbm4scale/CBM-IPDPS25.git --recursive
   cd CBM-IPDPS25/
   docker build -t cbm4gnn .
   ```
3. **Inside the Docker Container**  
   Once inside the container, navigate to the project directory and set up the environment:
    ```bash
    cd CBM-IPDPS25/
    python setup.py
    export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
    export PYTHONPATH=./:$PYTHONPATH
    ```
## Running the Code

To reproduce the results of the paper, run the following scripts from `CBM-IPDPS25/`:

### `./scripts/alpha_searcher.sh`
This script calculates the execution time of the matrix multiplication method defined in `cbm/cbm4mm.py` via `benchmark/benchmark_matmul.py` for each combination of alpha values specified in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array. 

Upon completion, the script generates a results file named `alpha_searcher_results.txt`, which records the matrix multiplication runtime, in seconds, for each combination of alpha values and datasets using the CBM format. Additionally, the resulting file includes the execution time of the matrix multiplication method from `cbm/mkl4mm.py`, which converts the datasets to CSR format and serves as the baseline for comparison.

#### How to Run:
1. Open the `scripts/alpha_searcher.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
      
2. Run `./scripts/alpha_searcher.sh` inside the `CBM-IPDPS25/` direction.

### `./scripts/compression_metrics.sh`
This script evaluates the performance of matrix compression into the CBM format using `cbm/cbm4mm.py`. Specifically, it measures the time required to convert a matrix to CBM format and calculates the compression ratio relative to the CSR format for each combination of alpha values defined in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array.  

Upon completion, the script generates a results file named `compression_metrics_results.txt`, which records the compression time, in seconds, and the achieved compression ratio for each alpha value and dataset combination.

#### How to Run:
1. Open the `scripts/compression_metrics.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
      
2. Run `./scripts/compression_metrics.sh` inside the `CBM-IPDPS25/` direction.

### `./scripts/run_matmul.sh`
This script evaluates the performance of different matrix multiplication methods using both CBM and CSR formats located in:  
   - `cbm/cbm4mm.py`, `cbm/mkl4mm.py`, `cbm/cbm4ad.py`, `cbm/mkl4ad.py`, `cbm/cbm4dad.py`, and `cbm/mkl4dad.py`.

cbmusing `cbm/cbm4mm.py`. Specifically, it measures the time required to convert a matrix to CBM format and calculates the compression ratio relative to the CSR format for each combination of alpha values defined in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array.  

Upon completion, the script generates a results file named `compression_metrics_results.txt`, which records the compression time, in seconds, and the achieved compression ratio for each alpha value and dataset combination.

#### How to Run:
1. Open the `scripts/run_matmul.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
      
2. Run `./scripts/compression_metrics.sh` inside the `CBM-IPDPS25/` direction.



--------------------------------------------------------------------

# cbm-benchmark
Refactor of latest version of CBM4Scale. New lean and informative benchmark. (CHANGE)

## Setup
### with Conda
1. Install Intel oneAPI Base Toolkit locally as mentioned [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/overview.html)
2. Create a Conda environmnent:
```bash
conda create -n cbm python=3.11
conda activate cbm
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip uninstall numpy
pip install numpy==1.24.3
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install ogb
conda install cmake ninja wget prettytable scipy
```
3. Install repository:
```bash
git clone https://github.com/cbm4scale/CBM-IPDPS25.git --recursive
cd CBM-IPDPS25/
git submodule init
git submodule update
python setup.py # if Intel oneapi is not installed on your default dir use: --setvars_path PATH_TO_ONEAPI/setvars.sh
export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
export PYTHONPATH=./:$PYTHONPATH
```
### with Docker
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

## Running the code
To reproduce the results fom the article, you can run the following bash scripts:

### scripts/alpha_searcher.sh
Runs the matrix multiplication method of `cbm/cbm4mm.py` via `benchmark/benchmark_matmul.py` for each alpha value in ```ALPHAS=[...]``` 
and dataset in ```DATASETS=[...]```. At the end of the execution, the sript generates a new file `alpha_searcher_results.txt` which will
contain the running time (in seconds) of matrix multiplication using the CBM format for each combination of alpha values and datasets. 

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
