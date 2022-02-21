# Perron root bounds
Calculate bounds on the Perron (dominant) eigenvalue of nonnegative matrices.

Currently supports the bounds provided by:
- A. Melman: https://doi.org/10.1080/03081087.2012.667096
- F. Babouklis, M. Adam, N. Assimakis: DOI:10.1109/MACISE49704.2020.00016
- H. Lindqvist: https://doi.org/10.1016/S0024-3795(02)00314-2
- R. Bapat: https://doi-org.ezproxy.lib.umb.edu/2323199

# Use cases:
To quickly estimate/bound the dominant eigenvalue can be useful in:
- Topological entropy
- Spectral graph theory (graph properties such as connectivity, coloring, etc.)
- Predominant evolution of dynamical systems 
- Note: Bapat and Lindqvist are bounds of one matrix's Perron root, based on the Perron root (and corresponding eigenvectors) of another matrix

# TODO:
- Provide the example calculations from each paper
- Find and implement other bounds
- Catalog each bound's best use-cases
- Ensure Bapat and Lindqvist works for zero values (only tested with positive matrices)
