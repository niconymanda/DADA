# DADA
The increasing use of generative AI models to create realistic deepfakes poses significant challenges to information security, particularly in the realm of audio manipulation. This code addresses the pressing need for improved detection of audio deepfakes by proposing a novel approach that incorporates context awareness. 

## Authorship authentication model 
### How to build 
#### Prerequisites
`pip install -r requirements.txt`
#### How to run 
```bash
cd DADA
python authorship_attribution/main.py --model_name 'BERT-like model'
``` 
### Dataset 
The dataset used for training and testing is an augmented version of the original "In the Wild" dataset, supplemented with data from "WikiQuotes" as well as quotes, tweets, and interviews of all the public figures included in the original dataset. This augmentation was necessary as the initial dataset was not sufficiently large to effectively fine-tune a LLM. By incorporating a broader range of data sources, we ensured more comprehensive coverage of the linguistic styles, topics, and contextual variety needed for robust model performance. 
### Authorship Classification Model
This project fine-tunes a BERT-like model to perform authorship classification, using Triplet Loss to enhance separation between author embeddings. A linear classification head is added on top of the model to classify texts based on authorship. The model architecture is flexible and can utilize any Transformer-based model, such as BERT, RoBERTa, or DeBERTa, making it adaptable for various pre-trained encoders. The Triplet Loss setup encourages the model to group embeddings by author, helping improve the accuracy and robustness of author identification.
### TODOs

- [x] Contastive Loss
- [x] Triplet Loss
- [x] ABX test 
- [ ] Roberta
- [ ] Test T5
- [ ] Test BERT
- [ ] Test Deberta
- [x] Hyperparameter tuning (Optuna/RayTune)


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
