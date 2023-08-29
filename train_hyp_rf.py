
import yaml

from hororf.rf_trainer import run_train
from hororf.utils import set_seeds


def main():
    
    params_file = "params.yml"
    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Different seed for each of the 5 network embeddings
    params["seed"] = params["seed"] + params["class_label"]
    set_seeds(params["seed"])
    
    run_train(params, params_file)


if __name__ == "__main__":
    main()