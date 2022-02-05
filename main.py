from datasets.movielens import get_movielens_dataset
from evaluations.cross_validation import random_train_test_split
from evaluations.evaluation import mrr_score
from factorization.implicit import ImplicitFactorizationModel
from model_utils.layers import BloomEmbedding
from model_utils.networks import Net


def main():
    dataset = get_movielens_dataset(variant='100K')
    train, test = random_train_test_split(dataset)

    for hash_function in ('MurmurHash', 'xxHash', 'MD5', 'SHA1', 'SHA256'):
        print('*'*20, hash_function, '*'*20)
        user_embeddings = BloomEmbedding(dataset.num_users, 32, compression_ratio=0.4, num_hash_functions=2, hash_function=hash_function)
        item_embeddings = BloomEmbedding(dataset.num_items, 32, compression_ratio=0.4, num_hash_functions=2, hash_function=hash_function)

        network = Net(dataset.num_users, dataset.num_items, user_embedding_layer=user_embeddings, item_embedding_layer=item_embeddings)
        model = ImplicitFactorizationModel(n_iter=1, loss='bpr', batch_size=1024, learning_rate=1e-2, l2=1e-6, representation=network, use_cuda=False)
        model.fit(train, verbose=True)

        mrr = mrr_score(model, test)
        print(f'MRR Score: {mrr.mean()}')


if __name__ == '__main__':
    main()
