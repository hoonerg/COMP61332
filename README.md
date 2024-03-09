## Link to github Repository:
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
This code will download large files in the working directory (4.5GB) for using word2vec and pretrained model.
```sh
python setup.py
```

### User Inference
This is for inference where user can type sentence and get class prediction with pretrained model.
```sh
python user_infer.py
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

### Training / Inference
This is for training and testing model.
The code below will run training and test for each model.
```sh
python main.py SVM
python main.py LSTM
```
