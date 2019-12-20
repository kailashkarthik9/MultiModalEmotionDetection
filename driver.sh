###
# This is the main driver script to run the code for the project on Multi-Modal Emotion Detection
# Title - Transfer Learning for Emotion Recognition
# UNIs - ks3740, jyh2127, aa4461

# Abstract - Emotion recognition is a challenging task for multiple reasons – the abstract nature of human emotions,
# the context-dependent relationship between words in text and their conveyed emotion, the multi-modal nature in which
# humans exhibit and understand emotions, and the lack of large annotated datasets for training machine learning models.
# While most historic research on emotions has focused on unimodal data, recent work has focused on combining the
# modalities of text, speech and vision for this task. In this paper, we present a system that leverages text and
# speech data for emotion recognition. The issue of limited data availability is tackled using a transfer learning
# approach, both for text and speech. We fine-tune a pre-trained speaker identification model and text language model
# for this task and evaluate it on the IEMOCAP dataset.

# List of Tools - Kaldi, Scikit-Learn, HuggingFace Transformers
# List of Directories -
#   1. MultiModalEmotionDetection/ - Python Code for Text Processing and iterfacing with Kaldi
#   2. kaldi/  -  Kaldi code for LDA PLDA and pre-training and fine-tune TDNN
###

###
# Goals of this script
# 1. Create the base speaker recognition model trained on VoxCeleb2 data
# 2. Create the transfer learnt model for emotion detection fine-tuned on MELD data
# 3. Evaluate the fine-tuned model on IEMOCAP data
# 4. Extract xvectors from the model on MELD and IEMOCAP data
# 5. Create the fine-tuned BERT model trained on MELD and extract sentence embeddings for transcripts on MELD and IEMOCAP data
# 6. Create ark files containing multi-modal vector embeddings for the datasets
# 7. Run an LDA-PLDA pipeline to classify emotions
###

###
# RUN Amith's Code here
# Copy the created model to where Jessica's code needs to it be
# Make sure to cd back to the root directory
###

# Create the directories required for interfacing Kaldi with an external codebase
cd MultiModalEmotionDetection || exit
sh create_dirs.sh
cd ..

###
# Run Jessica's Script 1 - that generates the xvectors
# Make sure to cd back to the root directory
###

# Copy the created xvectors to the external code base's directory
cp -a kaldi/path_to_meld_xvectors/. MultiModalEmotionDetection/meld/data/kaldi/
cp -a kaldi/path_to_iemocap_xvectors/. MultiModalEmotionDetection/iemocap/data/kaldi/

# Run the text processing code to create fine-tuned BERT and multi-modal embedding ark files
echo "The follwing code expects that MELD and IEMOCAP datasets are downloaded and extracted into the following directories:"
echo "MultiModalEmotionDetection/meld/data and MultiModalEmotionDetection/iemocap/data"
cd MultiModalEmotionDetection || exit
sh run.sh
cd ..

# Copy the created ark files to kaldi for LDA-PLDA
### Jessica please make sure the casing in multimodal matches what you want in the kaldi/path but please don't change the source directory name
cp -a MultiModalEmotionDetection/meld/data/kaldi/Text/. kaldi/path
cp -a MultiModalEmotionDetection/meld/data/kaldi/Speech/. kaldi/path
cp -a MultiModalEmotionDetection/meld/data/kaldi/Multimodal/. kaldi/path

cp -a MultiModalEmotionDetection/iemocap/data/kaldi/Text/. kaldi/path
cp -a MultiModalEmotionDetection/iemocap/data/kaldi/Speech/. kaldi/path
cp -a MultiModalEmotionDetection/iemocap/data/kaldi/Multimodal/. kaldi/path

###
# Run Jessica's Script 2 - that performs LDA PLDA
# Make sure to cd back to the root directory
###