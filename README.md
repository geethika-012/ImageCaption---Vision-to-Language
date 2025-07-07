This project implements an image captioning system that generates natural language descriptions for input images using deep learning. It leverages a ResNet-101 encoder, an LSTM decoder with attention, and beam search decoding to produce high-quality captions.

 Overview
Image captioning is the task of generating a textual description from an image. It combines computer vision and natural language processing techniques. This project uses a CNN-RNN architecture:

Encoder: A pretrained CNN (ResNet-101) extracts image features.

Decoder: An LSTM with attention generates captions from the encoded features.

Beam Search: Improves caption quality by exploring multiple possible word sequences.

 Model Architecture
Encoder: Modified ResNet-101 (without final classification layer)

Decoder: LSTM-based decoder with:

Bahdanau-style attention

Gating mechanism (f_beta) to modulate attention

Embedding layer and fully connected output

Beam Search: Maintains top-k probable sequences during generation

Setup Instructions
1. Clone the repository
git clone https://github.com/<geethika-012>/imagecaption.git
cd imagecaption
2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt


File Structure

imagecaption/
│
├── models.py              # Encoder & Decoder models
├── caption.py             # Inference script
├── utils/                 # (Optional) Utilities
├── img/                   # Folder to store input images
├── BEST_checkpoint_*.pth.tar  # Trained model checkpoint
├── WORDMAP_*.json         # Word-to-index map
└── README.md              # Project readme
Run Inference
Make sure you have:

A sample image (e.g., img/dog.jpg)

A trained model (.pth.tar)

A word map JSON (.json)

Then run:
python caption.py \
  --img img/dog.jpg \
  --model BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar \
  --word_map WORDMAP_coco_5_cap_per_img_5_min_word_freq.json
 Output


Example output:

==================================================
GENERATED CAPTION:
a group of dogs standing on top of a dirt field
==================================================


Resources Used
PyTorch

TorchVision

Microsoft COCO Dataset (for pretraining)

ResNet-101 (ImageNet pretrained weights)

