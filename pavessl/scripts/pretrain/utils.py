import pickle
from argparse import Namespace


def check_run(experiment_args: Namespace) -> Namespace:
    args_file_path = experiment_args.results_dir / "run.args"
    try:
        with args_file_path.open("rb") as args_file:
            return pickle.load(args_file)
    except FileNotFoundError:
        with args_file_path.open("wb") as args_file:
            pickle.dump(obj=experiment_args, file=args_file, protocol=5)
        return experiment_args
