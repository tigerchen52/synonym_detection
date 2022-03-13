# Imputing Out-of-Vocabulary Embedding with LOVE Makes Language Models Robust with Little Cost
This is a Pytorch implementation of our paper 

## Environment setup
Clone the repository and set up the environment via "requirements.txt". Here we use python3.6. 
```
pip install -r requirements.txt
```
## Data preparation
In our experiments, we use the FastText as target vectors [1]. [Downlaod](https://fasttext.cc/docs/en/english-vectors.html).
After downloading, put the embedding file in the path `cl_oov_solver/data/` 

## Training
First you can use `-help` to show the arguments
```
python train.py -help
```
Once completing the data preparation and environment setup, we can train the model via `train.py`.
We have also provided sample datasets, you can just run the mode without downloading.
```
python train.py -dataset data/wiki_100.vec
```

## Evaulation
To show the intrinsic results of our model, you can use the following command and 
we have provided the trained model we used in our paper. 

```
python evaluate.py
```


## Reference
[1] Bojanowski, Piotr, et al. "Enriching word vectors with subword information." Transactions of the Association for Computational Linguistics 5 (2017): 135-146.


