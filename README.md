# Recommendation Systems -  Final Project

## Getting Started
Recommendation algorithms tend to have large input and output dimensionalities that dominate their overall size.

This makes them difficult to train, due to the limited memory of graphical processing units, and difficult to deploy on mobile devices with limited hardware.

To address these difficulties, we propose Bloom embeddings, a compression technique that can be applied to the input and output of neural network models dealing with sparse high-dimensional binary-coded instances.
 
#### Motivation
We would like to evaluate Bloom embeddings with different hash functions and compare them against these alternative methods.
 
#### Advantages of Bloom embeddings:
* They are computationally efficient and do not seriously compromise the accuracy of the model.
* They do not require any change to the core model architecture or training configuration.
* "On-the-fly" constant-time operation.
* Zero or marginal space requirements.
* Training time speedups.


## Installations
* Install [miniconda](https://conda.io/miniconda.html) distribution of python3 using this [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Add conda to your bashrc:

```bash
echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
```
* Create and activate your virtual environment:
```
conda env update -f environment.yml      # create the virtual environemnt
conda env list                           # make sure the environemnt was added
conda activate recommandation-systems    # activate it
```

## Datasets
[MovieLens](https://paperswithcode.com/dataset/movielens) 

## References
* Published article: [Getting deep recommenders fit: Bloom embeddings for sparse binary input/output networks](https://arxiv.org/abs/1706.03993)

## Authors
* Sharon Mordechai.
* Eran Krakovsky.