Image Classifier (TensorFlow/Keras)

A TensorFlow/Keras image classifier built on MobileNetV2 and trained on the Oxford Flowers 102 dataset, with a command-line prediction tool.

Overview

An image classification project built with TensorFlow/Keras, using a MobileNetV2-based model trained on the Oxford Flowers 102 dataset to classify flower species from images. Built as the capstone project of a Udacity AI Programming with Python nanodegree.

Features
Image preprocessing pipeline (resizing, normalization) for feeding images into the model
Transfer learning on top of a pretrained MobileNetV2 backbone
Model training and evaluation workflow
predict.py — a command-line script that takes an image path and returns the predicted flower class (with confidence)
Tech

Python, TensorFlow, Keras, MobileNetV2, NumPy.

Usage
bash
python predict.py /path/to/image.jpg --top_k 3

(Adjust flags to match your actual predict.py argument names once uploaded.)

Context

Final project for the Udacity "AI Programming with Python" Nanodegree.

Note: if you'd rather present this repo as a different classification task (e.g. a separate dog-detection project), let me know and I'll rewrite this to match the actual code you upload.
