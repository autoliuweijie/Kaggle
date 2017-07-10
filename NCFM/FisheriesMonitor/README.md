# FisheriesMonitor


## Introduction

This is Kaggle Competetion [The Nature Conservancy Fisheries Monitoring](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring)


## Directories

The usage of these directories is

	dataset/  # store dataset
	cache/  # store some intermediate result
	resluts/  # store final reslut
	project/  # the main part of this project


## Create Environment

### Create virtualenv

We will use python2.7. Assume you are at the directory of this project "FisheriesMonitor/". 

```
$ virtualenv .env
```

### Install Packages

```
$ source .env/bin/activate
$ pip install -r requirements.txt
```

### Import OpenCV API to Virtualenv

We use opencv3.1 and python2.7, make sure the opencv3.1 is installed in your machine. My opencv API file is at

	/usr/local/lib/python2.7/dist-packages/cv2.so

So I import the API file cv2.so to virtualenv

```
$ ln -s /usr/local/lib/python2.7/dist-packages/cv2.so .env/local/lib/python2.7/site-packages/cv2.so
```

### Best Submssion

#### Step 1:

Train inception v3 on manual_dataset_balanced by "train_ipv3_on_manual_dataset.ipynb" if the model has not been trained.

#### Step 2:

Predict by "predict_on_manual_dataset.ipynb" and create results on 

    results/manual_region_balanced/ipv3_manual_region_balanced_t19.csv

#### Step 3:

Train inception v3 on original dataset by "training_inception_v3.py"

#### Step 4:

Predict by "predict_with_inception_v3.ipynb"

#### Step 5:

Merge results by "merge_result_full_with_manual_region.ipynb"


















