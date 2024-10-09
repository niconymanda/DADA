# DADA
The increasing use of generative AI models to create realistic deepfakes poses significant challenges to information security, particularly in the realm of audio manipulation. This code addresses the pressing need for improved detection of audio deepfakes by proposing a novel approach that incorporates context awareness. 

## Authorship authentication model 
### How to build 
#### Prerequisites
`pip install -r requirements.txt`
#### How to run 

### Dataset 
The dataset used for training and testing is an augmented version of the original "In the Wild" dataset, supplemented with data from "Brainy Quotes," as well as quotes, tweets, and interviews of all the public figures included in the original dataset. This augmentation was necessary as the initial dataset was not sufficiently large to effectively fine-tune a large language model (LLM). By incorporating a broader range of data sources, we ensured more comprehensive coverage of the linguistic styles, topics, and contextual variety needed for robust model performance. 
### Results
#### Results on Test set
| Model  | Specs  | Accuracy | F1 Score | Presicion | Recall | 
| ------------- | ------------- |  ------------- | ------------- | ------------- |  ------------- |
| RoBERTa  base  | e: 20, b_s: 32  | 0.18  | 0.16  | 0.18  | 0.15  | 
| RoBERTa  base  | ...  | ...  | ...  | ...  | ...  | 
| RoBERTa large | ...  | ...  | ...  | ...  | ... |
| T5 | ...  | ...  | ...  | ...  | ... |
...

#### Results on spoofed set
| Model  | Specs  | Accuracy |  Presicion | Recall | F1 Score |
| ------------- | ------------- |  ------------- | ------------- | ------------- |  ------------- |
| RoBERTa  base  | e: 20, b_s: 32  | 0.62  | 0.63  | 0.62  | 0.60  | 
| RoBERTa large | ...  | ...  | ...  | ...  | ... |
| T5 | ...  | ...  | ...  | ...  | ... |
...
