# deep-AMPpred
## Introduction
* Antimicrobial peptides (AMPs) are small peptides that play an important role in disease defense. As the problem of pathogen resistance caused by the misuse of antibiotics intensifies, the identification of AMPs as alternatives to antibiotics has become a hot topic. Accurately identifying AMPs using computational methods has been a key issue in the field of bioinformatics in recent years. Although there are many machine learning-based AMP identification tools, most of them do not focus on or only focus on a few functional activities. Predicting the multiple activities of antimicrobial peptides can help discover candidate peptides with broad-spectrum antimicrobial ability. We propose a two-stage AMP predictor deep-AMPpred, in which the first stage distinguishes AMP from other peptides, and the second stage solves the multi-label problem of 13 common functional activities of AMP. deep-AMPpred combines the ESM-2 model to encode the features of AMP and integrates CNN, BiLSTM and CBAM models to discover AMP and its functional activities. The ESM-2 model can efficiently capture the global contextual features of peptide sequences, while the combination of CNN, BiLSTM and CBAM utilizes the local feature extraction ability of CNN, the long-term and short-term dependency modeling ability of BiLSTM and the attention mechanism of CBAM, making feature recognition more comprehensive and improving the performance of deep-AMPpred in the prediction of AMP and its multiple functional activities. Experimental results show that deep-AMPpred has achieved good performance in performance indicators such as the accuracy of AMP and its functional activity recognition. It further proves the effectiveness and significance of using the ESM-2 method to capture effective feature information of peptide sequences and combining multiple deep learning models to identify AMP and its functional activities.


## Environment
* python 3.9.19
* biopython 1.83
* numpy 1.24.4
* pandas 2.2.1
* scikit-learn 1.4.1.post1
* scipy 1.12.0
* torch 1.13.0+cu117
* tqdm 4.66.5
* transformers 4.44.2
## Usage
* Please download the relevant files of the esm2_t12_35M_UR50D pre-trained model on Hugging Face and put them in the created Rostlab/esm2_t12_35M_UR50D folder.
* We provide pred.py under the prediction folder, we support independent prediction for each category. Just modify the file address in the code to the input data you are interested in and the model saved for the corresponding category.
