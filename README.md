# CBM-IPDPS25
This repository contains the refactored implementation of the code used for the experimental evaluation in **"Accelerating Graph Neural Networks Using a Novel Computation-Friendly Matrix Compression Format"**, accepted at **IPDPS 2025**.  

The artifacts in this repository are also part of our nomination to the **Best Open-Source Contribution Award Submission** of **IPDPS 2025**.  

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
   python conda_setup.py  # If Intel oneAPI is not installed in the default directory, use: --setvars_path PATH_TO_ONEAPI/setvars.sh
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
   docker run --rm -ti --ipc=host --name cbm4gnn_instance cbm4gnn /bin/bash
   ```
3. **Inside the Docker Container**  
   Once inside the container, navigate to the project directory and set up the environment:
    ```bash
    cd CBM-IPDPS25/
    python docker_setup.py
    export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
    export PYTHONPATH=./:$PYTHONPATH
    ```
## Reproducing the Experiments from the Paper

### `./scripts/alpha_searcher.sh`
This script calculates the execution time of the matrix multiplication method defined in `cbm/cbm4mm.py` via `benchmark/benchmark_matmul.py` for each combination of alpha values specified in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array. 

Upon completion, the script generates a results file named `results/alpha_searcher_results.txt`, which records the matrix multiplication runtime, in seconds, for each combination of alpha values and datasets using the CBM format. Additionally, the resulting file includes the execution time of the matrix multiplication method from `cbm/mkl4mm.py`, which converts the datasets to CSR format and serves as the baseline for comparison.

> **Note:** `cbm/cbm4mm.py` and `cbm/mkl4mm.py` contain python classes to store matrix **A** in CBM and CSR format, and support matrix products of the form **A** @ **X**.
> Here, **A** is the adjacency matrix of the dataset and **X** is a dense real-valued matrix. 


#### How to Run:
1. Open the `scripts/alpha_searcher.sh` and modify the following variables:
   - `MAX_THREADS=...`  
        Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
        Include in this array the thread counts you want to experiment with.
      
2. Run `./scripts/alpha_searcher.sh` inside the `CBM-IPDPS25/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.

   - `ALPHAS=(...)`  
       Include in this array the alpha values to be considered.

### `./scripts/compression_metrics.sh`
This script evaluates the performance of CBM's compression algorithm using `cbm/cbm4mm.py` via `benchmark/cbm_construction`. Specifically, it measures the time required to convert a matrix to CBM format and calculates the compression ratio relative to the CSR format for each combination of alpha values defined in the `ALPHAS=[...]` array and datasets in the `DATASETS=[...]` array.  

Upon completion, the script generates a results file named `results/compression_metrics_results.txt`, which records the compression time, in seconds, and the achieved compression ratio for each alpha value and dataset combination.

#### How to Run:
1. Open the `scripts/compression_metrics.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
      
2. Run `./scripts/compression_metrics.sh` inside the `CBM-IPDPS25/` direction.

Other configuration options (use default values to reproduce our experiments):   
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
     
   - `ITERATIONS=(...)`  
        Include in this array the number of times dataset should be converted to CBM format..

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.

   - `ALPHAS=(...)`  
       Include in this array the alpha values to be considered.


### `./scripts/matmul.sh`
This script evaluates the performance of different matrix multiplication methods with both CBM and CSR formats using:  
   - `cbm/cbm4{mm,ad,dad}.py` and `cbm/mkl4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.
   - The alpha values used to convert the dataset to CBM format are defined in `benchmark/utilities.py`.

Upon completion, the script generates a results file named `results/matmul_results.txt`, which records time related metrics for matrix multiplication.

> **Note:** `cbm/cbm4ad.py` and `cbm/mkl4ad.py` contain python classes to store matrix **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 

> **Note:** `cbm/cbm4dad.py` and `cbm/mkl4dad.py` contain python classes to store matrix **D^(-1/2)** @ **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **D^(-1/2)** @ **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 


#### How to Run:
1. Open the `scripts/matmul.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  

2. Run `./scripts/matmul.sh` inside the `CBM-IPDPS25/` direction.  

Other configuration options (use default values to reproduce our experiments):    
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.



### `./scripts/inference.sh`
This script evaluates the performance of the CBM format in the context of Graph Convolutional Neural Network (GCN) inference:  
- The graph's laplacian is represented in CBM (`cbm/cbm4dad}.py`) or CSR (`cbm/mkl4dad}.py`) formats.
- The inference itself is executed by `benchmark/benchmark_inference.py`.  
- The alpha values used to convert the dataset to CBM format are defined in `benchmark/utilities.py`.

Upon completion, the script generates a results file named `results/inference_results.txt`, which records the time related metrics for GCN inference.

#### How to Run:
1. Open the `scripts/inference.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
       
2. Run `./scripts/inference.sh` inside the `CBM-IPDPS25/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
        Include in this array the datasets that should be considered..  

   - `NUM_HIDDEN_LAYERS=(...)`  
        Include in this array the number of hidden layers to be added to the GCN.
   
   - `HIDDEN FEATURES=(...)`  
        Include in this array the number of columns to be used in the feature and learnable matrices.
     
   - `EPOCHS=(...)`  
        Include in this array the number of GCN inferences to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup epochs to be run before recording starts.

### `./scripts/validate.sh`
This script validates the correction different matrix multiplication methods with CBM formats using: 
- `cbm/cbm4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.

This validation is performed by comparing the resulting matrices (element-wise) between the classes mentioned above and `cbm/mkl4{mm,ad,dad}.py`.
Again, the alpha values used are the ones set in `benchmark/utilities.py`.

#### How to Run:
1. Open the `scripts/validate.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your machine.
   - `THREADS=(...)`  
     Include in this array the specific thread counts you want to experiment with.  
       
2. Run `./scripts/valiate.sh` inside the `CBM-IPDPS25/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `RTOL=...`  
        Specifies the relative tolerance interval to be considered during validation.

   - `ATOL=...`  
        Specifies the absolute tolerance interval to be considered in the validation.

## Aditional Artifacts  

If you would like to use the CBM format in your own pytorch projects, the classes mentioned before might be a bit restrictive since matrix **D** is always a diagonal degree matrix of **A**.
Instead, consider using the class defined in `cbm/cbm.py` which accepts two optional and precalculated diagonal matrices (`d_left` and `d_right`) as input. The script 
`benchmark/validate_generic.py` exemplifies the usage of `cbm/cbm.py`.
<!-- 
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
```-->
