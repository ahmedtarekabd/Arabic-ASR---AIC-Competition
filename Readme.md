# Automatic Speech Recognition (ASR) System with LAS Model

This repository contains an implementation of an Automatic Speech Recognition (ASR) system using the Listen, Attend, and Spell (LAS) model. The system is designed to convert Arabic speech to text, leveraging deep learning techniques and state-of-the-art neural network architectures.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Methodologies](#methodologies)
4. [Technical Details](#technical-details)
5. [Reproducibility](#reproducibility)
6. [Performance Assessment](#performance-assessment)
7. [Usage Instructions](#usage-instructions)
8. [Dependencies](#dependencies)
9. [Acknowledgements](#acknowledgements)
10. [Contact](#contact)

## Introduction

This ASR system aims to accurately transcribe Arabic audio into text. The model follows the Listen, Attend, and Spell (LAS) architecture, which uses an encoder-decoder mechanism with attention for efficient and effective speech recognition.

## System Architecture

The ASR system is built on the Listen, Attend, and Spell (LAS) model, consisting of three main components:

1. **Listener (Encoder)**: Converts audio features into a high-level representation.
2. **Attender (Attention Mechanism)**: Aligns the high-level representation with the output sequence.
3. **Speller (Decoder)**: Generates the text transcription from the aligned features.

### Diagram of the LAS Model

```plaintext
Input Audio -> [Listener] -> High-Level Features -> [Attender] -> Context Vectors -> [Speller] -> Output Text
```

### Detailed Architecture

1. **Listener**:
   - Multiple layers of bi-directional LSTMs.
   - Takes Mel-frequency cepstral coefficients (MFCCs) as input.
   - Outputs down-sampled feature representations.

2. **Attender**:
   - Mechanism to calculate attention weights over the encoded features.
   - Generates context vectors for each output step.

3. **Speller**:
   - A stack of LSTM layers.
   - Produces one character at a time based on the context vector and previous characters.
   - Contains a fully connected layer with softmax activation to predict the next character.

## Methodologies

1. **Preprocessing**:
   - Audio files are converted to MFCC features.
   - Transcriptions are tokenized into character sequences.

2. **Training**:
   - The model is trained using the Connectionist Temporal Classification (CTC) loss.
   - Teacher forcing is used to stabilize training by feeding the correct output back into the model.

3. **Inference**:
   - During inference, beam search decoding is employed to generate the most likely transcription.

4. **Evaluation**:
   - Word Error Rate (WER) and Character Error Rate (CER) are used to evaluate the performance of the model.

## Technical Details

### Model Training

- **Dataset**: The model was trained on a dataset consisting of Arabic speech recordings and their corresponding transcriptions.
- **Batch Size**: 16
- **Learning Rate**: 0.001 with an Adam optimizer.
- **Epochs**: 50
- **Training Time**: Approximately 20 hours on a single GPU.

### Model Checkpointing

- **Checkpointing**: We used the `ModelCheckpoint` callback to save the best model based on validation loss.
- **Resuming Training**: The system can resume training from any checkpoint to avoid loss of progress due to interruptions.

### Inference

- **Decoder**: The model employs a beam search decoder for generating text from audio inputs.
- **Vocabulary**: The system uses a vocabulary size of 52 characters, including Arabic letters and special tokens.

## Reproducibility

To reproduce the results and use the system for your own data, follow these steps:

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ASR-LAS-Model.git
cd ASR-LAS-Model
```

### Step 2: Prepare the Data

- Place your audio files in a directory named `audio/`.
- Ensure each audio file has a corresponding transcription in `transcripts.csv` in the format `audioID,transcript`.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model

```bash
python train.py --config=config.yaml
```

### Step 5: Infer Transcriptions for New Audio Files

```bash
python infer.py --input_folder=path/to/your/audio --output_csv=transcriptions.csv
```

### Step 6: Evaluate the Model

Evaluate the model's performance using the evaluation script:

```bash
python evaluate.py --model_path=path/to/best_model.pth --test_data=path/to/test_data
```

## Performance Assessment

The model was evaluated on a test set of Arabic speech recordings. The following metrics were used:

- **Word Error Rate (WER)**: 15.2%
- **Character Error Rate (CER)**: 4.5%

These metrics indicate the model's accuracy in transcribing Arabic speech.

## Usage Instructions

1. **Training**: Use `train.py` with the appropriate configurations in `config.yaml` to train the model.
2. **Inference**: Use `infer.py` to transcribe new audio files and save the transcriptions to a CSV file.
3. **Evaluation**: Use `evaluate.py` to calculate WER and CER on the test set.

## Dependencies

- Python 3.8+
- PyTorch 1.8+
- torchaudio 0.8+
- NumPy
- pandas
- tqdm
- librosa

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Acknowledgements

This project is based on the original LAS model architecture proposed by Chan et al. and has been adapted for Arabic speech recognition. We also thank the authors of the PyTorch library for providing an efficient platform for building and training deep learning models.

## Contact

For any questions or issues, please contact the project maintainer:

- **Team**: 3AM
- **Name**: Ahmed Tarek
- **GitHub**: [ahmedtarekabd](https://github.com/ahmedtarekabd)
