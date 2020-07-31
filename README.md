## Cross-Domain Emotion Detection using Transfer of Multi-modal Embeddings

### Authors : Kailash Karthik S and Amith Ananthram

#### kailashkarthik.s@columbia.edu, amith.ananthram@columbia.edu

### BERT for Text Emotion

* Download the datasets DailyDialog, IEMoCap, MELD and Crema-D from their respective sources
* Run the formatting code for each dataset from `dataset/formatting`. This formats the dataset files into CSVs with our normalized emotion set.
* Run the BERT data preparation notebooks from `dataset/preparation`. This generates PyTorch dataset files that can be directly used for training.
* To fine-tune BERT models for emotion classification, use the notebook `fine-tuning/BERT for Emotion Detection` with the appropriate datasets and base models
* To generate embeddings using the fine-tuned models, run the notebooks in `embeddings`. Note that the restructuring notebook is essential for compatibility with the kaldi code

### Multi-Modal Emotion Detection

* Refer to `kaldi-plda/` for instructions and code for the downstream multimodal task.

### Pre-trained Models

* Pre-trained models on combinations of the four datasets mentioned above can be found [here](https://drive.google.com/drive/folders/1XS6wpWurD9m6LO350DSY_G30qhJFhU25?usp=sharing). Request access with introduction and valid reason.
* Text embeddings for the datasets trained on combinations of the four datasets can be found [here](https://drive.google.com/drive/folders/1qETbB5XswS5edzhRFybcxr32W9EDuw_m?usp=sharing).