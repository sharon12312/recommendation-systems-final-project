from experiments.experiment import run_experiment
from experiments.plot import plot_experiment_results


def main():
    run_experiment(save_path='./results')
    plot_experiment_results(save_path='./results')


if __name__ == '__main__':
    main()
