 ImageCaption - Vision to Language

**ImageCaption** is a deep learning project that transforms visual data into natural language descriptions. This model is capable of generating human-like captions for images using a Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) with attention mechanisms.


 Overview

This project implements an image captioning pipeline trained on the MS-COCO dataset. It takes an image as input and generates a descriptive sentence. It combines computer vision and natural language processing, bridging the gap between visual understanding and language.

Key features:
- Image encoder using **ResNet101**.
- Decoder with **LSTM + Attention** mechanism.
- Beam Search for generating high-quality captions.
- Easy-to-run inference script with custom images.

 Demo
Example: Input Image: (dog.jpg)
Generated Caption: "a group of dogs standing on top of a dirt field"

You can try this on your own images by following the usage steps below.

 Project Structure :
ImageCaption---Vision-to-Language/
├── caption.py # Main inference script
├── models.py # Encoder and Decoder model classes
├── utils/ # Utility functions and image processing
├── data/ # Folder to place test images
├── checkpoints/ # Pretrained .pth.tar model files
├── word_maps/ # Word map (JSON file for vocabulary)
├── README.md # This file



Setup Instructions :

1. Clone the repository
git clone https://github.com/geethika-012/ImageCaption---Vision-to-Language.git
cd ImageCaption---Vision-to-Language

2. Create a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt
If a requirements.txt file is missing, install the essentials:
pip install torch torchvision pillow

Usage
1. Download the pretrained model
Due to GitHub size limits, the pretrained .pth.tar file is not uploaded. You can download it from this external source:
https://drive.google.com/file/d/1dOeLBwExkqD-cCQLRuaKqRkTY9JABEPZ/view?usp=drive_link
Place the file in the root directory of the project.

2. Run image captioning
python caption.py \
  --img img/dog.jpg \
  --model BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar \
  --word_map WORDMAP_coco_5_cap_per_img_5_min_word_freq.json

Model Architecture
Encoder
Uses pretrained ResNet-101 from torchvision.
Extracts feature vectors from the image.
The final convolutional layer's output is flattened and passed to the decoder.

Decoder with Attention
Embedding layer followed by LSTM-based sequence decoder.
Bahdanau-style attention mechanism focuses on relevant parts of the image for each word prediction.
Outputs tokens one at a time until <end> is reached.

Beam Search
Beam size configurable with --beam_size.
Maintains multiple hypotheses while decoding, improving caption quality.

Limitations
The model is trained on MS-COCO; it might not generalize well to very different domains.
Struggles with abstract objects or rare scenarios.
Generated captions may be grammatically correct but semantically inaccurate in rare cases.





