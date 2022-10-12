# How-to Guides

doc2data is currently structured along two top-level modules and the subpackage `experimental` .

The top-level modules contain functionality to create PDF collections and to access contents of individual files. The modules are:

Module | Description
--------------- | -------------
pdf | Reading PDF files and creating collections
utilities | Utilities

The subpackage `doc2data.experimental` contains modules for feature creation and model training:

Module | Description
--------------- | -------------
preprocessing   | Feature extractors for images, tokens and embeddings
ocr             | Wrapper for the dotTR OCR package
base_processors | Data pipelines for model training with TensorFlow & PyTorch
task_processes  | Task-specific processors
trainers        | Generic model training
utils           | Additional utilities

## Colab Notebooks

The following notebooks showcase typical applications of the above modules. The starting point is a number of PDF files with annotations. The output is a neural network for a specific document processing task.

Colab | Notebook | Description
--------------- | ------------- | -------------
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vtQWQbEuNqbigEpYLDELe32Vf6GJfYO6?usp=sharing) | Document classification | Train a custom Keras model to classify pages as images on a subset of the RVL-CDIP dataset.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mr02PfO4mDZ7t96ASOj1C6Vmf-BDoKE-?usp=sharing) | Token classification | Fine-tune the LayoutXLM model in PyTorch to classify each token on a document page.






