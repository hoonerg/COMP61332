## Link to the code's github Repository:
https://github.com/hoonerg/COMP61332

## How to run the code

### Setup
Run the following to install a subset of necessary python packages for our code
```sh
conda create -n [name] python==3.11
conda activate [name]
pip install -r requirements.txt
```

### Dependencies
This code will download large files in the working directory.
The first file is googlenews-vectors-negative300.bin.gz (1.5GB) for using word2vec.
The second file is pretrained SVM model for inference.
```sh
python setup.py
```

### Inference
This is for inference where user can type sentence and get class prediction with pretrained model.
```sh
python infer.py
```
This will show you user input form four times.
Type sentence, model type (SVM or LSTM), the first entity, and the second entity.
```sh
Enter the sentence: Sentence here without quotation marks.
Enter Model Type (SVM/LSTM): Either SVM or LSTM.
Enter the first entity: First entity here. Can be a word or group of words
Enter the second entity: Second entity here. Can be a word or group of words
```

## How to train the models

### Data preprocessing
This will generate preprocessed csv files from raw data files (XML).
```sh
python processing.py
```

### Training
This is for training and testing model.
The code below will run training and test for each model.
```sh
python main.py SVM
python main.py LSTM
```
