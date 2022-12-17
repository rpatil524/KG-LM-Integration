# KG-LM-Integration
This is the implementation of **K**nowledge **I**nte**G**rated BERT (KIG-BERT) proposed by Aisha and Xiangru for CS 848 (Knowledge Graph) course project.

## Paper
Find our paper with training and evaluation details in [link-to-paper](https://github.com/tanny411/KG-LM-Integration/blob/master/Language_Model_Knowledge_Graph_Integration.pdf)

**Abstract**: Recent developments in large language modeling have greatly accelerated the performances of NLP applications. Yet they remain largely dependent on their training data and thus prone to being factually inaccurate and socially biased. It is hard to correct the models after the fact due to their large size requiring high compute and large amounts of supervised training data. This paper proposes a minimal compute, no-pretrain framework for improving language model factual accuracy by incorporating knowledge graph information. Unlike human-written text, facts in knowledge graphs like Wikidata are accurate and free from bias. Comparison with baselines shows that our methods have promise in making language models factually accurate as well as retaining language understanding. We also build a facts dataset to test our work using template sentences and Wikidata entities to further evaluate the proposed system. 

## Datasets
1. [Linked Wikitext-2](https://rloganiv.github.io/linked-wikitext-2): A dataset that connects spans of text to Wikidata entities.
2. [Facts Dataset](https://github.com/tanny411/KG-LM-Integration/blob/master/generate_test_data/sythetic_dataset.jsonl): A dataset consisting of fact-sentences generated using templates and Wikidata entities collected with SPARQL queries.

## Usage
All the experimental results can be reproduced by the jupyter notebook KIG-Bert.ipynb. Detailed documentation and instruction is in the notebook.
 
## Requirements  
1. GPU is needed for training and evaluation.
2. Git Large File Storage package is needed. Please find the intrustion on the installing it in [installing-git-large-file-storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)
3. Python version: `3.10`
4. Install pip packages with `pip3 install -r requirements.txt`
