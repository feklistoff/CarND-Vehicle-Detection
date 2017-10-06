## Vehicle Detection Project
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
<img src="writeup_images/overview.jpg" width="500px">

[This](https://github.com/udacity/CarND-Vehicle-Detection) Udacity's repository contains starting files for the Project.

My detailed solution **[writeup](https://github.com/feklistoff/udacity-carnd-project4/blob/master/Writeup_Project_5.md)**.

### The Project
---
The goals/steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Train a Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected

Files:
* `vehicle_det_final.ipynb` - Final pipeline notebook.
* `building_pipeline.ipynb` - Building pipeline steps and visualization.
* `out_video.mp4` - Processed video
* `clf_scaler.p` - Saved trained classifier and scaler
* `coeffs.p` - Saved calibration parameters for lane detection

This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

or Python 3.5 and the following libraries installed:

* [Jupyter](http://jupyter.org/)
* [NumPy](http://www.numpy.org/)
* [Open CV](http://opencv.org/)
* [scikit-image](http://scikit-image.org/)
* [scikit-learn](http://scikit-learn.org/)
* [SciPy](https://www.scipy.org/)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.
