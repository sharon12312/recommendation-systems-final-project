from datasets.movielens import get_movielens_dataset
from evaluations.cross_validation import random_train_test_split
from evaluations.evaluation import mrr_score
from factorization.implicit import ImplicitFactorizationModel


def main():
    dataset = get_movielens_dataset(variant='100K')
    train, test = random_train_test_split(dataset)
    model = ImplicitFactorizationModel(n_iter=3, loss='bpr')
    model.fit(train, verbose=True)
    mrr = mrr_score(model, test)


if __name__ == '__main__':
    main()
