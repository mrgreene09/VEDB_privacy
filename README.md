# Privacy Script for Visual Experience Dataset (VEDB)
Detect and blur faces using [RetinaFace](https://github.com/serengil/retinaface)

## Installation
First, clone this repository:
```shell
git clone git@github.com:mrgreene09/VEDB_privacy.git
```

We recommend setting up a dedicated Anaconda environment for this code. 
```shell
conda env create -f privacyEnvironment.yml
conda activate privacyScript
```

## Use
Edit line 20 of the Python script to point to a csv file containing the video names to be processed. GPU acceleration is highly recommended.
