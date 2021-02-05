# Software Underground Rendezvous
## Importance of noise estimates when solving inverse problems in the "real" world
---

This repository contains the material for the Feb 5th 2021.
This is the starting point for an ongoing discussion and is in no way a finished product.

### Summary document

The discussion will follow this [article]()

### Conda Setup

To create the conda environment for this tutorial run:

```
conda env create -f environment.yml
```

The environment is called `rdz-sean`.

### Contents of repo

```
rdz-sean
│   README.md
│   LICENSE
│   environment.yml
│   rv-noise_est_pt1.ipynb
│   rv-noise_est_pt2.ipynb
│   .gitignore
│
└───scripts
│   │   dc_inv_utils.py
│
└───data
│   │   rv_dipole_dipole_dc.txt
│   │   rv_dipole_dipole_topo.txt
│
└───results
    │   setup.pkl
    │   inv1.pkl
    │   inv2.pkl
    │   inv3.pkl  
```

The two Jupyter notebooks contain the code to produce the results (pt1) and produce the images used in the discussion (pt2).

The scripts directory has the python helper functions used to setup, invert, save, load and plot the results.

The data directory contains the data used in the example.

The results directory contains the pickled output of pt1. These are loaded in pt2 for plotting.


