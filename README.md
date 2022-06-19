# TraceReconstruction-v2
This project consist in the creation of a simple trace reconstruction program for physical purposes. From a Monte Carlo simulation a list of rectilinear trace are created and made interact with a detector to generate the point of interaction. After being adequately saved using a proper format the Hought transformation is used on GPU to reconstruct the traces. The reconstructed traces are then compared to the originals..

The detector is bidimensional with three sensitive plates. Each plate is discretized into pixels.

The MonteCarlo simulation is made in such a way as to always guarantee at least an hit on the first plate of the detector, but it could be configured also to always garantee an hit on every plate.

The implementation of the Hough transformation is made in cuda and it is implemented in two different ways:
- using Thrust
- with a custom implementation of the calculation on the GPU, defining the core necessary, allocationg and freeing memeory, etc...

What was observed is that the first implementation has a speed advantace compared to the second.

The data for the analisy is plotted using ROOT.

The version of C++ used is C++17
The version of NVCC used is 10.2
