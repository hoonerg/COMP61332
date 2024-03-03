import sys
from train import main as training_main
from test import predict
import pandas as pd

pd.set_option('display.width', 1000)

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > else None

    training_main(model_type)
    predict(model_type)
