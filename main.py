from datasets.movielens import get_movielens_dataset


def main():
    dataset = get_movielens_dataset(variant='100K')
    print('hello')


if __name__ == '__main__':
    main()
