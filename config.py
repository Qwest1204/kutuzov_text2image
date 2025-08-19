import cv2
import torch
from math import log2
from sentence_transformers import SentenceTransformer

START_TRAIN_AT_IMG_SIZE = 8
DATASET = 'large_random_10k'
MODEL_EMBEDDER = model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = "mps" if torch.mps.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 8, 8, 4, 4, 4, 4, 4]
CHANNELS_IMG = 3
Z_DIM = 64  # should be 512 in original paper
IN_CHANNELS = 64  # should be 512 in original paper
CRITIC_ITERATIONS = 4
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(16, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4