# Motion Planning Assignment - Behavior Cloning

Thao Dang, HS Esslingen

This repository contains necessary files for behavior cloning in with the car racing environment in gymnasium. It is intended as an introductory assignment in the Motion Planning course.

Most material here is based on Andreas Geiger's class Self-Driving Cars at the University of Tübingen.

The assignment shall be run in colab. The main files for this are

- the notebook: MP_Assignment_Behavior_Cloning.ipynb
- the utility functions: utils.py
- the data files: demos_no_braking.npz, demos_with_braking.npz, demos_recovery.npz
- optional script for recording your own data: record_demonstrations.py

All other files are for development only.

To install a local environment, use this setup:

```bash
conda env create -f environment.yml
conda activate behavior-cloning
python -m ipykernel install --user --name=behavior-cloning
```