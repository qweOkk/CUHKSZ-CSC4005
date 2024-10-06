#!/bin/bash
#SBATCH -o ./Project1-PartC-pics.txt
#SBATCH -p Project
#SBATCH -J Project1-PartC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:1

# Necessary Environment Variables for Triton
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/ptxas                                                                      
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/cuobjdump                                                              
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvdisasm  
export PATH=/opt/rh/rh-python38/root/usr/bin:$PATH
# Get the current directory
CURRENT_DIR=$(pwd)/src/scripts

# Sequential PartC (Array-of-Structure)
echo "Sequential PartC Array-of-Structure (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/sequential_PartC_aos ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_aos.jpg
echo ""

# Sequential PartC (Structure-of-Array)
echo "Sequential PartC Structure-of-Array (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/sequential_PartC_soa ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_soa.jpg

echo ""

# SIMD PartC

echo "SIMD(AVX2) PartC (Optimized with -O0)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/simd_PartC_O0 ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_simd_O0.jpg
echo ""

echo "SIMD(AVX2) PartC (Optimized with -O2)"
srun -n 1 --cpus-per-task 1 ${CURRENT_DIR}/../../build/src/cpu/simd_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_simd.jpg
echo ""

# MPI PartC
echo "MPI PartC (Optimized with -O2)"
for num_processes in 32
do
  echo "Number of processes: $num_processes"
  srun -n $num_processes --cpus-per-task 1 --mpi=pmi2 ${CURRENT_DIR}/../../build/src/cpu/mpi_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_mpi.jpg
  echo ""
done

# Pthread PartC
echo "Pthread PartC (Optimized with -O2)"
for num_cores in 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/pthread_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_pthread.jpg ${num_cores}
  echo ""
done

# OpenMP PartC
echo "OpenMP PartC (Optimized with -O2)"
for num_cores in 32
do
  echo "Number of cores: $num_cores"
  srun -n 1 --cpus-per-task $num_cores ${CURRENT_DIR}/../../build/src/cpu/openmp_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_openmp.jpg ${num_cores}
  echo ""
done

# CUDA PartC
echo "CUDA PartC"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/cuda_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_cuda.jpg
echo ""

# OpenACC PartC
echo "OpenACC PartC"
srun -n 1 --gpus 1 ${CURRENT_DIR}/../../build/src/gpu/openacc_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_openacc.jpg
echo ""

# Triton PartC
echo "Triton PartC"
srun -n 1 --gpus 1 python3 ${CURRENT_DIR}/../gpu/triton_PartC.py ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral_triton.jpg
echo ""