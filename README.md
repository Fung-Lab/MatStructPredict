# MatStructPredict: An Open Source Library for GNN-Powered Structure Prediction

MatStructPredict is a machine learning library that offers simple, flexible pipelines for structure prediction and active learning.

## Table of Contents
- [MatStructPredict: An Open Source Library for GNN-Powered Structure Prediction](#matstructpredict-an-open-source-library-for-gnn-powered-structure-prediction)
    - [Table of Contents](#table-of-contents)
    - [Motivation](#motivation)
    - [Features](#features)
    - [Installation](#installation)
    - [Quick Start: Structure Prediction](#quick-start-structure-prediction)
    - [Contributing](#contributing)
    - [License](#license)
    - [Citation](#citation)


## Motivation

With more powerful and more accurate Graph Neural Networks coming into play, structure prediction using GNNs has become a fast and effective method for generating structures at scale. There have been multiple occasions where a vast amount of structures were generated using GNNs for property optimizaiton. However, creating programs to run GNN-based structure prediction requires specialized knowledge of PyTorch and Machine Learning. To help enable people of varying levels of machine learning knowledge to generate structures, we have created MatStructPredict.

MatStructPredict is a library that offers simple, customizable pipelines for structure prediction. The library offers the following features:
- Training and Evaluating ML Models
- Composition generation
- Global Optimization
  - BasinHopping
- Structure Prediction
    - Optimize structures for multiple objectives
    - Optimize atomic positions and atomic cell

By simplifying the process of predicting structures, MatStructPredict gives researchers the ability to generate structures for their own use cases, regardless of whether or not they are familiar with machine learning.

## Features

- **Pre-trained Model Support**: Use multiple pre-trained models for ASE optimization:
  - Chgnet
  - MACE
  - M3GNet
 
- **MatDeepLearn Model Features**: Use all models supported by MatDeepLearn for:
  - Training
  - Evaluating
  - Batch Optimization
  - Custom Objective Structure Prediction

- **Flexible Property**: Support for various molecular and materials properties:
  - Energy prediction
  - Force prediction (both conservative and non-conservative)
  - Stress tensor prediction
    
- **Flexible Objectives**: Support for various molecular and materials properties:
  - Energy
  - Novelty
    - Embedding Distance
    - Uncertainty
  - LJR Loss

- **Structure Prediction**: Pipelines for Structure Prediction from start to finish:
  - SMACT Valid Composition generation
    - Custom compositions
    - Random Lithium Compositions
    - Random Generic Compositions
  - Global Optimization with Basin Hopping
    - Includes following perturbs:
      - Cell
      - Positions
      - Atomic Numbers
      - Add/remove/swap atoms
  - Saving structures
  - Finetuning model on new structures

## Installation
TODO
```bash
pip install matstructpredict
```

## Quick Start: Structure Prediction

Use example.py and mdl_config.yml for a quick structure prediction run using a MatDeepLearn model. Remember to adjust the file paths to your corresponding dataset and ideal save paths.

Example.ipynb provides a Jupyter Notebook that takes users through each step of the Structure Prediction process.

## Contributing
 TODO
## License
TODO
## Citation
TODO
If you use MatStructPredict in your research, please cite:

TODO
```bibtex
```
