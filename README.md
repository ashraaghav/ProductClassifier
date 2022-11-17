# ProductClassifier

Creates a model that predicts the Product's category based on 
the description and manufacturer. The task has 3 main steps:
* Clean the data descriptions and generating the embeddings 
using pretrained BERTModel from huggingface
* Build an SVM to use the embeddings to identify the product category
* Use FastAPI to serve the model via HTTP REST API.

## Setup & Usage

Setup a new environment and install the dependencies listed in 
`requirements.txt` using `pip` as follows:

```bash
pip install -r requirements.txt
```

The python scripts to execute the modeling steps can be found in `scripts/`.


***STEP 1:*** Generate embeddings for the dataset
```
python scripts/generate_embeddings.py --datadir ./data --savedir ./results
```

***STEP 2:*** Train classifier
```
python scripts/train_model.py --datadir ./data --savedir ./results
```

***STEP 3:*** Serve the model via HTTP API (FAST API)
```
python main.py
```

Unittests can be found in the folder `tests/`.

## Performance 

The classifier achieves a test accuracy of **~99.7%** on the given 
products dataset. A more detailed step-by-step analysis is provided 
in the jupyter notebook `Product Category Classifier.ipynb`

## Deployment & Future steps

While this is just a simple HTTP API, to make things more production friendly, 
the next step is to use a Docker container so that the model can 
be deployed seamlessly.

Once the model goes to production, the model building and re-training procedure
should be more Pipeline oriented, so that each step is cleaner and more modular.
