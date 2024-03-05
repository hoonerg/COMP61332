import sys
from train import main as training_main
from test import predict
import pandas as pd

pd.set_option('display.width', 1000)

def get_user_input():
    sentence = input("Enter the sentence: ")
    first_entity = input("Enter the first entity: ")
    second_entity = input("Enter the second entity: ")
    
    formatted_input = f'"{sentence}", "{first_entity}", "{second_entity}"'
    return formatted_input

if __name__ == "__main__":
    user_input = get_user_input()
    
    #predict("Predicting with SVM", user_input)
    
    predict("Predicting with LSTM ", user_input)
