# LDhelmet_parallel

## TODO
- Add openmp to the makefile
- Add command line option to take in arguments num MCMCs, temperatures, how often to propose swap
- Handle MCMC chains in the rjmcmc layer. Define the shared data structures here
- Implement the swapping probability calculations (shouldn't be able to re-use 2 of the 4 likelihood evaluations I believe)
- Run experiments...I guess
