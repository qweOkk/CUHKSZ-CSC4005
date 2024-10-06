# Compile & Execute
Make sure under `project1` directory.
```
mkdir build && cd build
cmake ..
make
```
Then run the code under `project1` directory.
```
cd ..
sbatch src/scripts/sbatch_PartA.sh
sbatch src/scripts/sbatch_PartB.sh
sbatch src/scripts/sbatch_PartC.sh
```
Independently to run PartA/PartB/PartC and the you can get the performanc of each part. 

Run the code under `project1` directory.
```
sbatch src/scripts/sbatch_PartC_pics.sh
```
Then you can see the output pictures under the direcotry `project1/images`, and they are. ![Alt text](1728224108035.png)

# Explanation
## 1.Pthread
The Pthreads model in this program allows for parallel computation by dividing the image into smaller tasks that can be processed concurrently by multiple threads. The use of atomic operations for task allocation ensures that the threads do not interfere with each other, and the program efficiently utilizes the available CPU cores to speed up the image filtering process.
## 2.OpenMP
OpenMP is similar to the pthread model. It allows for parallel computation by dividing the image into smaller tasks that can be processed concurrently by multiple threads but automatically achieved by the openmp operation.

Both OpenMP and Pthread uses TLP(Thread Level Parallelism) to run multiple threads within a program concurrently.
## 3.MPI
MPI distributes the workload across multiple processes, allowing each process to work on a portion of the image simultaneously and use non-blocking communication to overlap computation and communication, reducing idle time.Moreover, MPI manages resources by balancing the load and utilizing multiple cpus, thereby maximizing the use of available computational power.  

Different from OpenMP and Pthread, MPI mostly uses PLP(Process Level Parallelism) to run multiple processes in parallel, with each process potentially running on a different CPU core or node.

## 4.SIMD 
SIMD, specifically AVX2,
vectorizes operations to process multiple data points simultaneously and reduces memory access by fetching and storing larger chunks of data. Moreover.SIMD reduces loop overhead by decreasing the number of iterations.