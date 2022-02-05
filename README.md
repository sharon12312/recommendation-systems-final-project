# Recommendation Systems -  Final Project

## Getting Started
**Recommendation algorithms** tend to have large input and output dimensionalities that dominate their overall size.
This makes them difficult to train, due to the limited memory of graphical processing units, and difficult to deploy on mobile devices with limited hardware.
To address these difficulties, we propose **Bloom embeddings**, a compression technique that can be applied to the input and output of neural network models dealing with sparse high-dimensional binary-coded instances.
 
### Motivation
We would like to evaluate Bloom embeddings with different hash functions and compare them against these alternative methods.

### Bloom embedding layers
Large embedding layers are a performance problem for fitting models. Although the gradients are sparse, PyTorch updates the entire embedding layer at every backward pass. Computation time is wasted on applying zero gradient steps to the whole embedding matrix.

To alleviate this problem, we can use a smaller underlying embedding layer, and probabilistically hash users and items into that smaller space. With good hash functions, collisions should be rare, and we should observe fitting speedups without a decreased inaccuracy.

### Advantages of the Bloom embeddings Layers:
* They are computationally efficient and do not seriously compromise the accuracy of the model.
* They do not require any change to the core model architecture or training configuration.
* "On-the-fly" constant-time operation.
* Zero or marginal space requirements.
* Training time speedups.

## Implementation
This implementation uses embedding layers indexed by hashed indices. Vectors retrieved by each hash function are averaged together to yield the final embedding. \
For hashing, we used different hash functions, and we performed hashing on the indices with a different seed, modulo the size of the compressed embedding layer. The hash mapping is computed once at the start of training and indexed into for every minibatch.
To exploit this sparsity structure, we simply skip zero entries in the input vector.


## Installations
* Install [miniconda](https://conda.io/miniconda.html) distribution of _python3_ using this [instruction](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
* Add _conda_ to your bashrc:

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
* [MovieLens](https://paperswithcode.com/dataset/movielens)
* [Recommender Datasets Repository](https://github.com/sharon12312/recommender-datasets)

## References
* RecSys 2017 paper: [Getting deep recommenders fit: Bloom embeddings for sparse binary input/output networks](https://arxiv.org/abs/1706.03993)

## Authors
* Sharon Mordechai.
* Eran Krakovsky.