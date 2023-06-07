import argparse
import json
import numpy as np
from pathlib import Path
import sys
import time

from src.nn.train import make_model, make_model_with_dropout, train_model, eval_model
import src.utils.helpers as helpers

def main(config_arg):
    config_file = Path(config_arg).absolute()
    if not config_file.is_file():
        raise Exception("Config file doesn't exist.")
    config = helpers.read_config(config_file)
    run_nn(config)

def run_nn(config):
    output_dir_path = Path(config["output"]["dir"]) # Where all the output is saved

    # 0. Redirect output
    if config["output"]["file"] is not None:
        output_file_path = output_dir_path.joinpath(config["output"]["file"])
        sys.stdout = open(output_file_path,'wt')

    # 1. Load data
    X_train = helpers.read_df(config["data"]["train"]["features"]).values.astype(np.float32)
    y_train = helpers.read_df(config["data"]["train"]["labels"]).values.astype(np.int).squeeze()
    X_val = helpers.read_df(config["data"]["val"]["features"]).values.astype(np.float32)
    y_val = helpers.read_df(config["data"]["val"]["labels"]).values.astype(np.int).squeeze()
    X_test = helpers.read_df(config["data"]["test"]["features"]).values.astype(np.float32)
    y_test = helpers.read_df(config["data"]["test"]["labels"]).values.astype(np.int).squeeze()

    # 2. Build model
    if config["model"]["dropout"]:
        model = make_model_with_dropout(config["model"]["layers"], config["model"]["lr"])
    else:
        model = make_model(config["model"]["layers"], config["model"]["lr"])

    # 3. Load weights
    weights_file = config["model"]["load_weights_file"]
    if weights_file is not None:
        weights_file_path = Path(weights_file).absolute()
        model.load_weights(str(weights_file_path))

    # 4. Train model
    if not config["model"]["skip_training"]:
        start_time = time.time()
        model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=config["model"]["epochs"], stop_early=config["model"]["stop_early"])
        time_elapsed = time.time() - start_time
        print("Training took: {0:.1f} seconds.\n".format(time_elapsed))

    # 5. Save model weights
    weights_file = config["model"]["save_weights_file"]
    if weights_file is not None:
        weights_file_path = output_dir_path.joinpath(weights_file).absolute()
        model.save_weights(str(weights_file_path))

    # 6. Save training history
    if not config["model"]["skip_training"] and config["model"]["save_history"]:
        helpers.write_pickle(history.history, output_dir_path.joinpath("history.pickle"))

    # 7. Evaluate model
    eval_model(model, X_train, y_train, text="Train data")
    eval_model(model, X_val, y_val, text="Val data")
    eval_model(model, X_test, y_test, text="Test data")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Specify config file')
    args = parser.parse_args()
    main(args.config)