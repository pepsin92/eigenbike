# eigenbike

This repository serves for FMFI UK machine learning course project.

## Installation

To install project dependencies, simply run
```pipenv install
```

## Execution

To run the project after installing all the necessary dependencies, simply execute either `cnn.py` or `svm.py`. As tensorflow is still unavailable on python 3.7 as of submission, please use theano: `KERAS_BACKEND=theano`.

On first run, please uncomment data scraping from either file (function `scrape`). It is currently commented to prevent unneccessary download of the dataset.
