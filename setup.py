import os
import subprocess
import argparse


def set_mkl_var_and_run_cbm_setup(setvars_path):
    # Change directory to where setup.py is located
    setup_dir = os.path.dirname(os.path.abspath(__file__)) + "/cbm"
    os.chdir(setup_dir)

    # Combine source and setup.py execution into one command
    combined_command = f"source {setvars_path} && python setup.py build_ext --inplace" # develop
    subprocess.check_call(combined_command, shell=True, executable="/bin/bash")


def cmake_for_arbok():
    # Change directory to where setup.py is located
    setup_dir = os.path.dirname(os.path.abspath(__file__)) + "/arbok"
    os.chdir(setup_dir)

    # Combine source and setup.py execution into one command
    cmake_command = "[ -d 'build' ] || cmake -B build -S . -DCMAKE_BUILD_TYPE=Release && cmake --build build/"
    subprocess.check_call(cmake_command, shell=True, executable="/bin/bash")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Python extension with environment variables sourced from Intel's setvars.sh.")
    parser.add_argument("--setvars_path",
                        type=str, default="/opt/intel/oneapi/setvars.sh",
                        help="The path to the setvars.sh script to source Intel environment variables (default: /opt/intel/oneapi/setvars.sh)")
    args = parser.parse_args()
    cmake_for_arbok()
    set_mkl_var_and_run_cbm_setup(args.setvars_path)
