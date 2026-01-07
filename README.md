# Analyzing BART Generalizability to Resume Summarization
This is the final project for CS 333: Natural Language Processing by Karen Xiao.
This project examines the performance of the `facebook/bart-large-cnn` model on the 
`burberg92/resume_summary` dataset available on Huggingface.

## Files
This project is organized into two Jupyter Notebooks:
- `evaluate.ipynb`: contains functions to evaluate the performance of the `bart-large-cnn` model on the resume dataset using the Huggingface pipeline API
- `finetune.ipynb`: performs evaluation and fine-tuning on the `bart-large-cnn` model

## How to Run
To run `evaluate.ipynb`, open the file in either Google Colab or Jupyter and run all cells sequentially. The resume dataset is quite small and the pipeline API makes the code execution relatively fast.

To run `finetune.ipynb`, use the NCSA Delta cloud computing system.

1. Start a Jupyter session
- Navigate to the following link: https://openondemand.delta.ncsa.illinois.edu/pun/sys/dashboard 
- Log in to your Delta account
2. Create a Jupyter Notebook using the following settings:
- Name of account: *bftp-delta-gpu*
- Partition: *gpuA40x4*
- Number of CPUs: *8*
- Amount of RAM: *48G*
- Number of GPUs: *1*
- Leave the Working Directory blank; it will default to /scratch/bftp/<USERNAME>
    
    NOTE: It may take a couple of minutes for your request to become active.
3. Once your Jupyter session is active, upload `finetune.ipynb` and run all cells sequentially

## Dependencies

* `torch` - PyTorch deep learning framework
* `torch.utils.data.DataLoader` - PyTorch utility class used for efficient loading of data in batches
* `datasets` - Hugging Face library for loading the resume_summary dataset
* `transformers.BartForConditionalGeneration` - BART model for sequence-to-sequence generation
* `transformers.BartTokenizer` - BART tokenizer
* `sentence_transformers.SentenceTransformer` - Model for creating sentence embeddings
* `sentence_transformers.util` - Utility functions for calculating cosine similarity
* `rouge.Rouge` - Python library for computing ROUGE evaluation metrics
* `matplotlib.pyplot` - Visualization library for model performance
* `numpy` - Python library for array operations