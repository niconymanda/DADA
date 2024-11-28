# DADA

The increasing use of generative AI models to create realistic deepfakes poses significant challenges to information security, particularly in the realm of audio manipulation. This work addresses the pressing need for improved detection of audio deepfakes by proposing a novel approach that incorporates context awareness. 


## Baselines

| Model      | Dataset    | Specs | Accuracy | F1 Score | EER |
| ---------- | ---------- | ----- | -------- | -------- | --- |
| AASIST []  | ASVSpoof21 | _     | ...      | ...      | ... |
| RawNet2 [] | ASVSpoof21 | _     | ...      | ...      | ... |
| SLIM []    | ASVSpoof21 | _     | ...      | ...      | ... |
The increasing use of generative AI models to create realistic deepfakes poses significant challenges to information security, particularly in the realm of audio manipulation. This code addresses the pressing need for improved detection of audio deepfakes by proposing a novel approach that incorporates context awareness. 

## Authorship authentication model 
### How to build 
#### Prerequisites
`pip install -r requirements.txt`
#### How to run 
To run the model training
```bash
cd DADA
python authorship_attribution/main.py --model_name 'BERT-like model'
``` 
How to run hyperparameter search tuning on a specific model.
You can define your hyperparameter space in the 'main_tune.py' file and use the 'config_tune' dictionary.
```bash
cd DADA
python authorship_attribution/main_tune.py --model_name 'BERT-like model'
``` 
### Dataset 
The dataset used for training and testing is an augmented version of the original "In the Wild" dataset, supplemented with data from "WikiQuotes" as well as quotes, tweets, and interviews of all the public figures included in the original dataset. This augmentation was necessary as the initial dataset was not sufficiently large to effectively fine-tune a LLM. By incorporating a broader range of data sources, we ensured more comprehensive coverage of the linguistic styles, topics, and contextual variety needed for robust model performance. 
### Authorship Classification Model
This project fine-tunes a BERT-like model to perform authorship classification, using Triplet Loss to enhance separation between author embeddings. A linear classification head is added on top of the model to classify texts based on authorship. The model architecture is flexible and can utilize any Transformer-based model, such as BERT, RoBERTa, or DeBERTa, making it adaptable for various pre-trained encoders. The Triplet Loss setup encourages the model to group embeddings by author, helping improve the accuracy and robustness of author identification.
### TODOs

- [ ] Implement test function for the spoofed data.
- [ ] Implement run training from the config file and save model params to config file.
- [x] Put the init of the optimizer, lr scheduler, loss function in the `Trainer` class
- [ ] Implement ~~triplet~~, ~~contrastive~~, ada_triplet, hinge, ~~cos2~~ loss
- [ ] Test with more encoders 
- [ ] Tests with cosine similarity in the Triplet loss instead of the L2 Norm 

## Repository Structure
  ```
 ├──  authorship_attribution
 │    └──config.py: Contains the default configuration settings for the model and training process.
 │    └──dataset.py: Handles data loading, tokenization, and preparation for training and evaluation.
 │    └──early_stopping.py: Implements the early stopping mechanism to prevent overfitting during model training.
 │    └──loss_functions.py: Defines custom loss functions used in the model training, such as Triplet Loss.
 │    └──main.py: The main script to run the model training.
 │    └──main_tune.py: Script to perform hyperparameter tuning on the model.
 │    └──model.py: Defines the architecture of the model, including the BERT-like model and the classification head.
 │    └──test_model.py: Script to evaluate the trained model on the test dataset.
 │    └──train.py: Contains the training loop and logic for training the model.
  ```

## Features
 * - **Trainer Class**: Contains the `Trainer` class responsible for training machine learning models on authorship attribution datasets using the custom Triplet Loss.
 * - **Tester Class**: Contains the `Tester` class for evaluating the performance of trained models on test datasets and the spoofed dataset.
 * - **Data Preprocessing**: Includes utilities for preprocessing text data, such as tokenization, vectorization, and normalization.
 * - **Model Evaluation**: Offers tools for evaluating model performance, including ABX accuracy, accuracy, precision, recall, F1-score, and confusion matrices.
 * - **Training Visulalization**: supports Wandb and Tensorboard
 * - **Visualization**: Supports various visualization techniques to help interpret model results and data distributions.
    * - **t-SNE Plotting**: Provides functionality to plot t-SNE on the modeled space after training the model.
    * - **Cosine Distribution Plots**: Plot cosine similarity distributions between the anchor and positive samples and anchor-negative samples.

### Results
#### Results on Test set
| Model  | Specs  | Accuracy | F1 Score | Presicion | Recall | 
| ------------- | ------------- |  ------------- | ------------- | ------------- |  ------------- |
| RoBERTa  base  | ...  | ...  | ...  | ...  | ...  | 
| RoBERTa  base  | ...  | ...  | ...  | ...  | ...  | 
| RoBERTa large | ...  | ...  | ...  | ...  | ... |
| T5 | ...  | ...  | ...  | ...  | ... |
...

#### Results on spoofed set
| Model  | Specs  | Accuracy |  Presicion | Recall | F1 Score |
| ------------- | ------------- |  ------------- | ------------- | ------------- |  ------------- |
| RoBERTa  base  | ...  | ...  | ...  | ...  | ...  | 
| RoBERTa large | ...  | ...  | ...  | ...  | ... |
| T5 | ...  | ...  | ...  | ...  | ... |
...
