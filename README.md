# conjugatePhaseRecon.jl

## B0-aware conjugate phase reconstruction on the GPU in Julia

- Based on work by Noll, Fessler and Sutton, and others [1]. 
- This should ideally become an implementation of the paper cited above, with adaptations to enable GPU compatibility, as well as an extension to multi-coil data.
- A demo of B0 correction on the GPU is provided, as well as a first stab at sensitivity-aware reconstruction and simulation functions. 
- Suggestions and improvements are very welcome! 

To run the demos, simply download and open the repository in the editor of your choice. Activate the environment contained within, and instantiate it as usual. Then, run the scripts.

The demos do require significant compute resources (>= 64GB RAM, >= 8GB VRAM) but should run reasonably quickly. Expect first compile to take a while, this is Julia after all :) 

[1]D. C. Noll, J. A. Fessler, and B. P. Sutton, “Conjugate phase MRI reconstruction with spatially variant sample density correction,” IEEE Trans. Med. Imaging, vol. 24, no. 3, pp. 325–336, Mar. 2005, doi: 10.1109/TMI.2004.842452.
