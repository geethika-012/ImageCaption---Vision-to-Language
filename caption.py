import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import json
import warnings
from models import Encoder, DecoderWithAttention

# Suppress warnings
warnings.filterwarnings("ignore")

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--img', '-i', required=True, help='Input image path')
parser.add_argument('--model', '-m', required=True, help='Trained model path (.pth.tar)')
parser.add_argument('--word_map', '-wm', required=True, help='Word map JSON path')
parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
args = parser.parse_args()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word map
with open(args.word_map, 'r') as f:
    word_map = json.load(f)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Fix Adam optimizer compatibility issue
print("Fixing Adam optimizer compatibility...")
import torch.optim as optim

# Monkey patch the Adam optimizer to handle missing defaults
original_adam_setstate = optim.Adam.__setstate__

def fixed_adam_setstate(self, state):
    if not hasattr(self, 'defaults'):
        self.defaults = {}
    self.defaults.setdefault("differentiable", False)
    self.defaults.setdefault("maximize", False)
    self.defaults.setdefault("foreach", None)
    self.defaults.setdefault("capturable", False)
    self.defaults.setdefault("fused", None)
    try:
        original_adam_setstate(self, state)
    except Exception as e:
        print(f"Warning: Could not restore Adam optimizer state: {e}")
        self.state = {}
        self.param_groups = []

optim.Adam.__setstate__ = fixed_adam_setstate

# Load model checkpoint
print("Loading model...")
try:
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    encoder = checkpoint['encoder']
    decoder = checkpoint['decoder']
    print("Encoder and decoder loaded successfully from checkpoint")
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    exit(1)

encoder = encoder.to(device)
encoder.eval()
decoder = decoder.to(device)
decoder.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Read and preprocess image
print(f"Loading image: {args.img}")
try:
    img = Image.open(args.img).convert("RGB")
    image = transform(img).unsqueeze(0).to(device)
    print(" Image loaded and preprocessed")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Generate caption
print("Generating caption...")
with torch.no_grad():
    encoder_out = encoder(image)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)

    encoder_out = encoder_out.view(1, -1, encoder_dim)
    num_pixels = encoder_out.size(1)

    k = args.beam_size
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    complete_seqs = []
    complete_seqs_scores = []

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)
        scores = torch.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores

        top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        k -= len(complete_inds)
        if k == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)

        if step > 50:
            break
        step += 1

# Generate final caption
if complete_seqs_scores:
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    sentence = ' '.join([rev_word_map[ind] for ind in seq 
                        if ind not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
    print("\n" + "="*50)
    print("GENERATED CAPTION:")
    print(sentence)
    print("="*50)
else:
    print("No complete sequences generated. Try reducing beam size or check your model.")
