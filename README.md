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
conda create --name behavior-cloning python=3.10.12 ipython
conda activate behavior-cloning
conda install conda-forge::gymnasium conda-forge::pygame numpy
conda install conda-forge::swig conda-forge::gymnasium-box2d
conda install conda-forge::moviepy
conda install pandas matplotlib
conda install pytorch torchvision -c pytorch
conda install anaconda::ipykernel anaconda::jupyter
python -m ipykernel install --user --name=behavior-cloning
```
