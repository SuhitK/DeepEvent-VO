![Build Failing](https://img.shields.io/badge/build-passing-brightgreen.svg)  ![Status Alpha](https://img.shields.io/badge/status-beta-blueviolet.svg)

# DeepEvent-VO

This is the project which [@epiception](https://github.com/epiception), [@gupta-abhay](https://github.com/gupta-abhay) and I worked on for the Robot Localization and Mapping (16-833) course project at CMU in Spring 2019. The motivation of the project is to see how [DeepVO](http://senwang.gitlab.io/DeepVO/) can be enhanced using event-frames. To read more about event cameras and event SLAM, see this [link](http://rpg.ifi.uzh.ch/research_dvs.html). Our reports and presentations can be found in the `reports` folder.

## Installation Instructions

This is a `PyTorch` implementation. This code has been tested on `PyTorch 0.4.1` with `CUDA 9.1` and `CUDNN 7.1.2`.

#### Dependencies

We have a dependency on a few packages, especially `tqdm`, `scikit-image`, `tensorbordX` and `matplotlib`. They can be installed using standard pip as below:

```
pip install scipy scikit-image matplotlib tqdm tensorboardX
```

To replicate the `python` environment that we used for the experimentation you can start a python virtual environment ([instructions here](https://docs.python-guide.org/dev/virtualenvs/)) and then run

```
pip install -r requirements.txt
````

The model assumes that FlowNet pre-trained weights are available for training. You can download the weights from [@ClementPinard's implementation](https://github.com/ClementPinard/FlowNetPytorch). Particularly, we need the weights for FlowNetS (flownets_EPE1.951.pth.tar). Instructions for downloadind the weights are in the README given there.

## Datasets

This model assumes the [MVSEC](https://daniilidis-group.github.io/mvsec/) datasets available from the [Daniilidis Group](https://daniilidis-group.github.io/) at [University of Pennsylvania](https://www.upenn.edu/). The code to sync the dataframes for event and intensity frames along with poses are available in the data folder.

## Running Code

Place the pre-processed data into each folder before running the models or change the `args.py` in each folder to accept data from a comman folder.

To run the code for base DeepVO results - without any fusion, from the base directory of the repository, run:

```
cd finetune/
sh exec.sh
```

To run the fusion model, from the base directory of the repository, run:

```
cd finetune/
sh exec_fusion.sh
```

Similarly models for the ablation experiments like scratch and freeze can be run by going to the respective folders.

## Acknowledgments
This code has been largely modified from DeepVO implementation by [krrish94](https://github.com/krrish94/DeepVO)
