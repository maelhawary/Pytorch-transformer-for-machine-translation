# Pytorch_Transformer_for_Machine_Translation
## Introduction
This is a PyTorch implementation of a Transformer-based deep learning model for langauge translation. The transformer model is inspired by the well-known Attention is All you need [https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html]. However, Two changes has been added to this architecture from the original one. The first modification is adding the norm layer before the attention model and before the position-wise neural network. the second modification is addeding a dropout layer after the output of the encoder block.
## Dataset
The transformer model is used for translating content from English to Italian. The dataset and the tokenizers are downloaded from Huggingface datasets. The source containing other langauges, so feel free to change the name in the configuration file

## Dependency
Python 3.8
Pytorch 2.1.0

## Train
$ python initial.py

