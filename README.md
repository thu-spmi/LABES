# LABES
This is the code for EMNLP 2020 paper "A Probabilistic End-To-End Task-Oriented Dialog Model with Latent Belief States towards Semi-Supervised Learning". [paper link]

## Requirements
- Python 3.6
- PyTorch 1.2.0
- NLTK 3.4.5

We use some NLP tools in NLTK which can be installed through:
```
python -m nltk.downloader stopwords punkt wordnet
```

## Data Preparation
1. Unzip raw data of [CamRest676](https://www.repository.cam.ac.uk/handle/1810/260970), [Stanford In-Car Assistant](https://nlp.stanford.edu/blog/a-new-multi-turn-multi-domain-task-oriented-dialogue-dataset/) and [MultiWOZ 2.1](https://www.repository.cam.ac.uk/handle/1810/294507), and also the GloVe word embeddings into the corresponding directories. Note that file "compressed_data_2.0.json.zip" is the raw MultiWOZ 2.0 data from [this repository](https://github.com/594zyc/damd-multiwoz), for normalizing entity names in the data preprocessing process. 

2. Data Preprocess
Raw data are preprocessed automatically during the first run of each dataset. See datasets.py and multiwoz_preprocess.py for what have been done in the data preprocessing process. 


## Running Experiments

### Training
```
python train.py -mode train -dataset [camrest|kvret|multiwoz] -method cvae -c spv_proportion=[a integer between 0-100] exp_no=your_exp_name
```

### Testing
```
python train.py -mode test -dataset [camrest|kvret|multiwoz] -method cvae -c eval_load_path=[experimental path]
```

## Reproducibility 
We release the models that obtain the best results in Table 1 and Table 2. Run the following commands for model evaluation.  
```
python train.py -mode test -dataset camrest -method cvae -c eval_load_path=experiments/camrest/camrest_best beam_search=True
python train.py -mode test -dataset kvret -method cvae -c eval_load_path=experiments/kvret/kvret_best beam_search=True
python train.py -mode test -dataset multiwoz -method cvae -c eval_load_path=experiments/multiwoz/multiwoz_best beam_search=True
```


## Bug Report
Feel free to create an issue or send email to zhangyic@umich.edu