import sys
from train import main as training_main
from test import predict
import pandas as pd

pd.set_option('display.width', 1000)

def get_user_input():
    # Prompt for user inputs
    sentence = input("Enter the sentence: ")
    first_entity = input("Enter the first entity: ")
    second_entity = input("Enter the second entity: ")
    
    # Format the user input as required
    formatted_input = f'"{sentence}", "{first_entity}", "{second_entity}"'
    return formatted_input

if __name__ == "__main__":
    user_input = get_user_input()
    
    #print("Predicting with SVM model...")
    #predict("SVM", user_input)
    
    print("\nPredicting with LSTM model...")
    predict("LSTM", user_input)
