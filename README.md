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
```sh
python setup.py
```

### Inference
The inference code use only LSTM model which is a final version.
```sh
python infer.py
```

### Training
There are two models
```sh
python main.py SVM
python main.py LSTM
```

### Data preprocessing
There are two models
```sh
python processing_suffix_x.py
```