# NCFM Fish Classification

This is Classification part of my project of [NCFM](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) in kaggle.


## 1. Directories

The usages of these directories are

    datasets/  # store dataset
    project/  # the main part of this project
    resluts/  # store final reslut


## 2. Create Environment

We will use virtualenv with python2.7 to build this project. Make sure the virtualenv is installed on your machine.

Assume that you are at the work directory of this project "NFCM_Classification/"

```
$ virtualenv .env
```
Install packages that will be used. if you are using a machine with Nvidia GPU and CUDA and cuDNN has been installed, you can use requirements_gpu.txt, otherwise use requirements.txt.

```
$ source .env/bin/activate
$ pip install -r requirements_cpu.txt
or
$ pip install -r requirements_gpu.txt
```

Import OpenCV API to Virtualenv. We use opencv3.1 and python2.7, make sure the opencv3.1 is installed in your machine. My opencv API file is at

    /usr/local/lib/python2.7/dist-packages/cv2.so

So I import the API file cv2.so to virtualenv

```
$ ln -s /usr/local/lib/python2.7/dist-packages/cv2.so .env/local/lib/python2.7/site-packages/cv2.so
```

Now you can start up the project by jupyter notebook or use the script below

```
$ ./start_jupyter_notebook.sh
```

## 3. Dataset

Move the origin dataset of this project offerd by competition sponsor to dataset/, and the directory structures is as follows

    dataset/
        |- origin_dataset/
            |- train/
                |- ALB/
                |- ...
                |- YFT/
                |- ALB.json
                |- ...
                |- YFT.json
            |- test_stg1/
            |- test_stg1.json

