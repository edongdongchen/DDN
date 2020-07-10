# Deep Decomposition Network (DDN)
This is the implementation of ECCV'20 paper: [Deep Decomposition Network for Inverse Imaging Problems](https://arxiv.org/abs/1911.11028). 
Code will be released soon.


By [Dongdong Chen](http://dongdongchen.com), [Mike E. Davies](https://scholar.google.co.uk/citations?user=dwmfR3oAAAAJ&hl=en).

The University of Edinburgh, UK.

### Table of Contents
0. [Keywords](#Keywords)
0. [Abstract](#Abstract)
0. [Requirement](#Requirement)
0. [Usage](#Usage)
0. [Citation](#citation)

### Keywords

Inverse problem, Deep learning, Physics consistence, Range-Nullspace decomposition 

### Abstract
Deep learning is emerging as a new paradigm for solving inverse imaging problems. However, the deep learning methods often lack the assurance of traditional physics-based methods due to the lack of physical information considerations in neural network training and deploying. The appropriate supervision and explicit calibration by the information of the physic model can enhance the neural network learning and its practical performance. In this paper, inspired by the geometry that data can be decomposed by two components from the null-space of the forward operator and the range space of its pseudo-inverse, we train neural networks to learn the two components and therefore learn the decomposition, i.e.,  we explicitly reformulate the neural network layers as learning range-nullspace decomposition functions with reference to the layer inputs, instead of learning unreferenced functions. We empirically show that the proposed framework demonstrates superior performance over recent deep residual learning, unrolled learning and nullspace learning on tasks including compressed sensing medical imaging and natural image super-resolution.

### Requirement
0. PyTorch >=1.0
0. CUDA >=8.5

### Usage

To Do...

### Citation

If you use these models in your research, please cite:

	@inproceedings{chen2019decomposition,
		author = {Chen, Dongdong and Davies, Mike E},
		title = {Deep Decomposition Learning for Inverse Imaging Problems},
		booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
		year = {2020}
	}
