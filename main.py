import sys
from train import main as training_main
from test import predict
import pandas as pd

pd.set_option('display.width', 1000)

if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else None
    user_input = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None

   #training_main(model_type)

    if len(sys.argv) > 2:
        predict(model_type, user_input)

    else:
        training_main(model_type)
        predict(model_type, user_input)
