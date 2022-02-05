import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.results import Results


def process_results(results, verbose=False):
    baseline = results.best_baseline()

    def like_baseline(x):
        for key in ('n_iter', 'batch_size', 'l2', 'learning_rate', 'loss', 'embedding_dim'):
            if x[key] != baseline[key]:
                return False

        return True

    data = pd.DataFrame([x for x in results if like_baseline(x)])
    best = (data.sort_values('test_mrr', ascending=False).groupby('compression_ratio', as_index=False).first())

    # Normalize per iteration
    best['elapsed'] = best['elapsed'] / best['n_iter']

    if verbose:
        print(best)

    baseline_mrr = (best[best['compression_ratio'] == 1.0]['validation_mrr'].values[0])
    baseline_time = (best[best['compression_ratio'] == 1.0]['elapsed'].values[0])
    compression_ratio = best['compression_ratio'].values
    mrr = best['validation_mrr'].values / baseline_mrr
    elapsed = best['elapsed'].values / baseline_time

    return compression_ratio[:-1], mrr[:-1], elapsed[:-1]


def plot_results(model, results, save_path):
    sns.set_style("darkgrid")

    for result in results:
        compression_ratio, mrr, elapsed = process_results(result, verbose=True)
        plt.plot(compression_ratio, mrr, label=result.get_filename())

    plt.ylabel("MRR ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs MRR ratio")

    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, f'_{model}_plot.png'))

    plt.close()

    for result in results:
        compression_ratio, mrr, elapsed = process_results(result)
        plt.plot(compression_ratio, elapsed, label=result.get_filename())

    plt.ylabel("Time ratio to baseline")
    plt.xlabel("Compression ratio")
    plt.title("Compression ratio vs time ratio")
    plt.legend(loc='lower right')

    plt.savefig(os.path.join(save_path, f'_{model}_time.png'))
    plt.close()


def plot_experiment_results(save_path='../results'):
    results = []
    for hash_function in ['MurmurHash', 'xxHash', 'MD5', 'SHA1', 'SHA256']:
        filename = f'{hash_function}_implicit_movielens_results.txt'
        filename_path = os.path.join(save_path, filename)
        results.append(Results(filename_path))

    plot_results('MovieLens', results, save_path)


if __name__ == '__main__':
    plot_experiment_results()
