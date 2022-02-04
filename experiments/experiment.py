import time
import numpy as np

from sklearn.model_selection import ParameterSampler
from datasets.movielens import get_movielens_dataset
from evaluations.cross_validation import random_train_test_split
from evaluations.evaluation import mrr_score
from experiments.results import Results
from factorization.implicit import ImplicitFactorizationModel
from model_utils.layers import BloomEmbedding, ScaledEmbedding
from model_utils.networks import Net
from model_utils.torch_utils import set_seed

CUDA = False
NUM_SAMPLES = 2
LEARNING_RATES = [1e-4, 5 * 1e-4]
LOSSES = ['bpr', 'adaptive_hinge']
BATCH_SIZE = [128, 256]
EMBEDDING_DIM = [32, 64]
N_ITER = [1, 2]
L2 = [1e-6, 1e-5]
COMPRESSION_RATIOS = [0.2, 0.4]


def set_hyperparameters(random_state, num):
    params = {
        'n_iter': N_ITER,
        'batch_size': BATCH_SIZE,
        'l2': L2,
        'learning_rate': LEARNING_RATES,
        'loss': LOSSES,
        'embedding_dim': EMBEDDING_DIM,
    }
    sampler = ParameterSampler(params, n_iter=num, random_state=random_state)

    for params in sampler:
        yield params


def build_factorization_model(hyperparameters, train, random_state):
    h = hyperparameters
    set_seed(42, CUDA)

    if h['compression_ratio'] < 1.0:
        item_embeddings = BloomEmbedding(train.num_items, h['embedding_dim'], compression_ratio=h['compression_ratio'], num_hash_functions=4, padding_idx=0)
        user_embeddings = BloomEmbedding(train.num_users, h['embedding_dim'], compression_ratio=h['compression_ratio'], num_hash_functions=4, padding_idx=0)
    else:
        item_embeddings = ScaledEmbedding(train.num_items, h['embedding_dim'], padding_idx=0)
        user_embeddings = ScaledEmbedding(train.num_users, h['embedding_dim'], padding_idx=0)

    network = Net(train.num_users, train.num_items, user_embedding_layer=user_embeddings, item_embedding_layer=item_embeddings)
    model = ImplicitFactorizationModel(loss=h['loss'], n_iter=h['n_iter'], batch_size=h['batch_size'], learning_rate=h['learning_rate'], embedding_dim=h['embedding_dim'], l2=h['l2'], representation=network, use_cuda=CUDA, random_state=random_state)

    return model


def evaluate_model(model, train, test, validation):
    start_time = time.time()
    model.fit(train, verbose=True)
    elapsed = time.time() - start_time

    print(f'Elapsed {elapsed}')
    print(model)

    test_mrr = mrr_score(model, test)
    val_mrr = mrr_score(model, test.tocsr() + validation.tocsr())

    return test_mrr, val_mrr, elapsed


def run(experiment_name, train, test, validation, random_state):
    results = Results(f'{experiment_name}_results.txt')

    best_result = results.best()
    if best_result is not None:
        print(f'Best result: {best_result}')

    # Find a good baseline
    for hyperparameters in set_hyperparameters(random_state, NUM_SAMPLES):
        hyperparameters['batch_size'] = hyperparameters['batch_size'] * 4
        hyperparameters['compression_ratio'] = 1.0

        if hyperparameters in results:
            print('Done, skipping...')
            continue

        print(f'==> Hyperparameters: {hyperparameters}')
        model = build_factorization_model(hyperparameters, train, random_state)
        test_mrr, val_mrr, elapsed = evaluate_model(model, train, test, validation)

        print(f'==> Test MRR: {test_mrr.mean()}, Validation MRR: {val_mrr.mean()}, Elapsed Time: {elapsed}')
        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)

    best_baseline = results.best_baseline()
    print(f'Best baseline: {best_baseline}')
    print('*'*200)

    # Compute compression results
    for compression_ratio in COMPRESSION_RATIOS:
        hyperparameters = best_baseline
        hyperparameters['compression_ratio'] = compression_ratio

        if hyperparameters in results:
            print('Compression computed')
            continue

        print(hyperparameters)
        model = build_factorization_model(hyperparameters, train, random_state)
        test_mrr, val_mrr, elapsed = evaluate_model(model, train, test, validation)

        print(f'Test MRR: {test_mrr.mean()}, Validation MRR: {val_mrr.mean()}, Elapsed Time: {elapsed}')
        results.save(hyperparameters, test_mrr.mean(), val_mrr.mean(), elapsed)
        print('*' * 200)

    return results


if __name__ == '__main__':
    random_state = np.random.RandomState(100)
    dataset = get_movielens_dataset('100K')
    test_percentage = 0.2

    train, rest = random_train_test_split(dataset, test_percentage=test_percentage, random_state=random_state)
    test, validation = random_train_test_split(rest, test_percentage=0.5, random_state=random_state)

    experiment_name = 'Implicit Factorization Model'
    run(experiment_name, train, test, validation, random_state)
