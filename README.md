# Fabrication-Aware Inverse Design of Nanophotonic Devices for 3D Laser-Nanoprinting

This project is associated with the manuscript of the same name (doi to follow). The repository can be used to reproduce all of the results of the manuscript.
It is suggested to run the optimization on a GPU, but especially the small metalens can be run on most hardware in a reasonable time.

First, we want to motivate the work and give a brief explanation of the steps we take in our optimization, especially how we actually model the fabrication process of 3D laser-nanoprinting.
Second, we highlight the main results of the manuscript.
Third, a quick guide on how to use the code (and how to run optimizations with different systems).

## Motivation
The goal of this work is to include a dosage accumulation into the optimization workflow.
When doing gradient-based optimization, typically only heuristic measures (such as Gauss-Filters) are used to include constraints imposed by the fabrication method.
However, there is no guarantee, that the imposed constraints realistically reflect the actual experimentally available design space.
Not only could this lead to non-fabricable designs, the optimization also cannot make use of fabrication-specific effects.

A simple Gaussian is not enough to capture the intricate shape of the high-NA laser focus spot used in 3D laser-nanoprinting, where the shape of a single basic unit (voxel) can differ significantly compared to a Gaussian shaped laser focus spot.
Additionally, in 3D laser-nanoprinting, the accumulated dose in the photoresist also plays a role.
In an effect known as the proximity effect, repeated illumination of the same voxel, where each individual illumination is below the polymerization threshold, can lead to polymerization of that voxel.
Lastly, the 3D laser-nanoprinting process itself also distorts the original design being used as a writing pattern.
Prediction and precompensation are necessary to obtain a desired printed design, which is often not trivial or easy to do.

## Direct-Laser-Writing Model
We incorporate a fully differentiable (programmed in torch) direct-laser-writing model into a nanophotonic inverse design workflow.
By calculating the point-spread-function (PSF) of the laser focus, we obtain a more accurate description of what a single voxel in our simulation would look like. 
Furthermore, we also model the dose accumulation and polymerization, allowing for the optimizer to make use of the proximity effect.
The parameters of the model include the experimental settings to obtain the PSF and how we use the PSF to obtain the final, printed design.
In particular, we make use of the laser power as our design parameter, normalizing other quantities such as polymerization threshold w.r.t. the laser power.
Most of the settings can also be chosen freely, if a different setup is required.
The magic happens in the dose accumulation model, not only allowing us to incorporate fabrication specific effects into our optimization, but also enabling us to optimize the writing pattern, which we can then use to predict the final, printed design all in one.

### Writing pattern $\sigma(\mathbf{r})$
Our choice of parametrization is density-based topology optimization. By parametrizing the relative permittivity using a density $0\le\rho\le 1$, we can obtain the optical response of said density.
However, we want to obtain a writing pattern as an intermediate step (which can be converted into an .stl (or similar) file).
This writing pattern serves as our starting point, which we (and the 3D laser-nanoprinting machine) will convert into a printed design.
To obtain the writing pattern, we simply binarize the density. We can omit all other requirements and constraints we would have, since only the printed design needs to abide to such restrictions.

### Dose Accumulation $D(\mathbf{r})
We model the dose accumulation in the photoresist $D(\mathbf{r})$ given the relative laser power $P_\text{laser}$, the writing pattern $\sigma(\mathbf{r})$ and the intensity of the PSF $I_\text{PSF}(\mathbf{r})$ as
$D(\mathbf{r}) = P^2_\text{laser} \cdot \sigma(\mathbf{r}) \cdot I^2_\text{PSF}(\mathbf{r})$.
Our controllable hyperparameter in the optimization is simply the relative laser power, taking over a similar role as the minimum feature size.

### Degree of polymerization $\tilde{\sigma}$ and final design
Finally, we convert the accumulated dose into the degree of polymerization $\hat{\sigma}(\mathbf{r})$ using a maximum degree of polymerization $\sigma_0$ and an experimental prefactor $c$
$\tilde{\sigma} = \sigma_0 \cdot ( 1 - \exp(-c\cdot D(\mathbf{r}))$.
The final design is then obtained by another binarization, assigning values above a polymerization threshold as material and as void otherwise.

## Further Considerations
We use the final design to calculate the optical response of the material distribution. We also consider structural integrity and robustness in our figure of merit, making the design fully 3D printable and more robust against experimental variations.
The optimization itself will start off at a low level of binarization and gets more and more binarized as we approach a local minimum. 


The entire workflow can be seen in the following image a), as well as a small metalens designed using this workflow in b).
<img width="1066" height="735" alt="first_pic" src="https://github.com/user-attachments/assets/e763ca9b-7eb5-48e5-9447-ddc80ffed2a3" />


## Results

We want to show some structures we designed using the DLW-Method. The upper image always shows the writing pattern and the bottom image shows the resulting structure once it was sent through the DLW-Model, representing the "printed" design.
<img width="2138" height="867" alt="structures" src="https://github.com/user-attachments/assets/354a1801-86d3-4994-a83e-5d69d5fd7e77" />


## Code Details
I apologize for the (somewhat messy) and lengthy code, I try to refactor it bit by bit when I have time. But simply running the ```main.py``` file reproduces the small metalens on a GPU (CPU is also possible, but then the torch tensors need to casted onto the CPU manually everywhere, I will fix that later).
Every optimization requires a "setting", which has to be put into ```settings/_setting_loader.py```, where the individual settings for the optimization are being specified (should ideally be a json, pull requests are always welcome for such an implementation. Pull requests are welcome in general.).
The setting can be used to set what kind of objective function should be optimized for, if Gauss filtering should be used, what projection should be used, if precompensation should be used or not, etc.
Every setting itself represents a specific run for a specific optimization problem. Each optimization problem is defined in its own ```_run.py``` file, which is being pulled by the ```dispenser.py``` file.
Technically speaking, one can set up their own ```_run.py``` file however they wish.
Each simulation problem consists off of a ```config_structure.py``` which defines the parameters of the simulation, an ```objective.py``` which defines the objective and a ```simulation.py``` file which assembles the simulation and returns the required quantities to calculate the objective.
```_objective_loader.py``` is then used to assemble the proper setting, e.g. with or without heat solver.

The DLW-Model can be found under ```filtering/dose_model``` as well as all of its settings. The projections are also implemented in their own sub-folder as well as the functions used to plot the final and intermediate results.
Extending all of the setups can be done by defining their own sub-routine and then using the ```_*_loader.py``` files to "distribute" them into the setting.

Lastly, the optimizers are set up to take gradients from jax. So wrappers are required to convert the gradients obtained otherwise into jax. The wrapper for the torch-based DLW-Model is implemented explicitly in the optimizers.

Since everything runs on a GPU, it is a bit difficult for me to get a definite requirement.yaml (or similar file) due to how things are set up on my working PC, but I will try to add one later.
Most importantly, all the standard libraries such as numpy, matplotlib, etc. should be installed, as well as torch and jax. Furthermore cupy and sympy are needed for the heat solver (or just remove tofea from everywhere).
