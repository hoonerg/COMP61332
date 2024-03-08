## How to run the code

### Setup
Run the following to install a subset of necessary python packages for our code
```sh
conda create -n [name] python==3.11
conda activate [name]
pip install -r requirements.txt
```

### Dependencies
You need to download googlenews-vectors-negative300.bin.gz (1.5GB) for word2vec in the working directory
The trained model for SVM (142MB) is added as a google drive link as well. By running the 
following command, the trained model will be downloaded to results/checkpoint/svm_best_model.pkl
```sh
python setup.py
```

### Inference
The inference code utilizes both LSTM and SVM model, depending on the model type provided.
```sh
python infer.py
```
This will show you user input form three times.
Type sentence, model type (SVM/LSTM), the first entity, and the second entity.
```sh
Enter the sentence: Sentence here without quotation marks.
Enter Model Type (SVM/LSTM): Either SVM or LSTM.
Enter the first entity: First entity here. Can be a word or group of words
Enter the second entity: Second entity here. Can be a word or group of words
```

## How to train the models

### Data preprocessing
This will generate preprocessed csv files from raw data.
```sh
python processing.py
```

### Training
There are two models available for training and test. 
The code below will run training and test for each model.
```sh
python main.py SVM
python main.py LSTM
```