import os
from pathlib import Path

files = {}

files["app/__init__.py"] = ""

files["app/config.py"] = """"""Application settings (override with environment variables)."""
import os
from pathlib import Path

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "1000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))
"""

files["app/common.py"] = """"""Constants, text preprocessing and vocabulary shared by training and serving."""
import json
import re
from collections import Counter

EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}

PAD_TOKEN, OOV_TOKEN = "<PAD>", "<OOV>"


def clean_text(text) -> str:
    """Lower-case, keep letters only and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class WordVocab:
    """Word-level vocabulary: index 0 = padding, index 1 = out-of-vocabulary."""

    def __init__(self, word2idx):
        self.word2idx = word2idx

    @classmethod
    def build(cls, texts, max_words):
        counter = Counter(word for text in texts for word in text.split())
        word2idx = {PAD_TOKEN: 0, OOV_TOKEN: 1}
        for word, _ in counter.most_common(max_words - 2):
            word2idx[word] = len(word2idx)
        return cls(word2idx)

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len):
        ids = [self.word2idx.get(word, 1) for word in text.split()][:max_len]
        return ids + [0] * (max_len - len(ids))

    def encode_batch(self, texts, max_len):
        return [self.encode(text, max_len) for text in texts]

    def to_json(self):
        return json.dumps(self.word2idx)

    @classmethod
    def from_json(cls, payload):
        return cls(json.loads(payload))
"""

files["app/models.py"] = """"""Recurrent text classifiers (shared by training and serving)."""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, units: int = 64):
        super().__init__()
        self.project = nn.Linear(hidden_dim, units)
        self.score = nn.Linear(units, 1, bias=False)

    def forward(self, hidden, mask):
        scores = self.score(torch.tanh(self.project(hidden))).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return context, weights


class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, cell="lstm", bidirectional=False, attention=False,
                 embed_dim=100, hidden_dim=128, dropout=0.4, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.as_tensor(embedding_matrix))
        self.embed_dropout = nn.Dropout(0.3)

        rnn_class = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_class(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)

        self.attention = AdditiveAttention(out_dim) if attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, input_ids, return_attention=False):
        mask = input_ids != 0
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.rnn(packed)

        weights = None
        if self.attention is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
            features, weights = self.attention(outputs, mask)
        else:
            last = hidden[0] if isinstance(hidden, tuple) else hidden
            features = torch.cat([last[-2], last[-1]], dim=1) if self.rnn.bidirectional else last[-1]

        logits = self.classifier(features)
        return (logits, weights) if return_attention else logits
"""

files["app/schemas.py"] = """"""Pydantic request / response models."""
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from .config import MAX_BATCH_SIZE, MAX_TEXT_CHARS


def _validate_text(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("text must not be empty")
    if len(value) > MAX_TEXT_CHARS:
        raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters")
    return value


class PredictRequest(BaseModel):
    text: str = Field(..., examples=["i feel so happy and grateful today"])

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        return _validate_text(value)


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, values: List[str]) -> List[str]:
        return [_validate_text(v) for v in values]


class EmotionPrediction(BaseModel):
    emotion: str
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictResponse(BaseModel):
    predictions: List[EmotionPrediction]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    version: str
    labels: List[str]
"""

files["app/predictor.py"] = """"""Model loading and inference for both supported backends."""
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .common import WordVocab, clean_text
from .config import ARTIFACTS_DIR

logger = logging.getLogger("emotion-api")


class EmotionPredictor:
    def __init__(self, artifacts_dir: Path):
        self.info = json.loads((artifacts_dir / "model_info.json").read_text())
        self.labels: List[str] = [self.info["labels"][str(i)] for i in range(len(self.info["labels"]))]
        self.max_len: int = self.info["max_len"]
        self.backend: str = self.info["backend"]
        self._lock = threading.Lock()

        if self.backend == "transformer":
            self._load_transformer(artifacts_dir)
        else:
            self._load_recurrent(artifacts_dir)

        self.predict(["warm up"])
        logger.info("Loaded model '%s' (%s backend)", self.info["name"], self.backend)

    def _load_transformer(self, artifacts_dir: Path) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = str(artifacts_dir / "best_model")
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self._proba = self._proba_transformer

    def _load_recurrent(self, artifacts_dir: Path) -> None:
        from .models import SequenceClassifier

        checkpoint = torch.load(artifacts_dir / "best_model.pt", map_location="cpu")
        self._model = SequenceClassifier(**checkpoint["config"])
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.eval()
        self._vocab = WordVocab.from_json((artifacts_dir / "vocab.json").read_text())
        self._proba = self._proba_recurrent

    def _proba_transformer(self, texts: List[str]) -> np.ndarray:
        encoded = self._tokenizer(
            texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self._model(**encoded).logits
        return torch.softmax(logits, dim=-1).numpy()

    def _proba_recurrent(self, texts: List[str]) -> np.ndarray:
        ids = torch.tensor(self._vocab.encode_batch([clean_text(t) for t in texts], self.max_len))
        with torch.no_grad():
            logits = self._model(ids)
        return torch.softmax(logits, dim=-1).numpy()

    def predict(self, texts: List[str]) -> List[Dict]:
        with self._lock:
            probabilities = self._proba(texts)
        outputs = []
        for row in probabilities:
            best = int(row.argmax())
            outputs.append(
                {
                    "emotion": self.labels[best],
                    "confidence": round(float(row[best]), 4),
                    "probabilities": {label: round(float(p), 4) for label, p in zip(self.labels, row)},
                }
            )
        return outputs


@lru_cache(maxsize=1)
def get_predictor() -> EmotionPredictor:
    return EmotionPredictor(ARTIFACTS_DIR)
"""

files["app/ui.py"] = """"""Gradio web interface for emotion classification."""
import gradio as gr
from .predictor import get_predictor


def classify(text: str):
    if not text or not text.strip():
        return {}
    return get_predictor().predict([text.strip()])[0]["probabilities"]


def build_demo() -> gr.Blocks:
    direct_examples = [
        ["i am feeling so wonderfully blessed and happy today"],
        ["i am so mad that my flight was canceled without any warning"],
        ["i felt a cold shiver run down my spine when i heard the sudden crash"],
        ["i miss my family so much and my heart feels completely broken"],
        ["i never expected to win this amazing award it is an unbelievable shock"],
        ["i adore being around you because you make every single day special"],
        ["i am really anxious about the big presentation tomorrow morning"],
        ["i am so grateful for all the kind support my friends showed me"],
        ["i felt deeply humiliated and furious when they laughed at my idea"],
        ["i was completely startled by the loud fireworks outside my window"],
    ]

    with gr.Blocks(title="Emotion Classifier") as demo:
        gr.Markdown(
            "# Emotion Classifier\n"
            "Type any English text. The model returns the dominant emotion and "
            "its confidence across all 6 classes."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(lines=5, label="Your text", placeholder="How are you feeling?")
                button = gr.Button("Analyze emotion", variant="primary")
            with gr.Column():
                output = gr.Label(num_top_classes=6, label="Predicted emotion")

        gr.Examples(examples=direct_examples, inputs=text, per_page=10)

        button.click(classify, inputs=text, outputs=output)
        text.submit(classify, inputs=text, outputs=output)
    return demo
"""

files["app/main.py"] = """"""FastAPI REST API with mounted Gradio UI."""
import logging
import time
from contextlib import asynccontextmanager

import gradio as gr
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .predictor import get_predictor
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EmotionPrediction,
    HealthResponse,
    PredictRequest,
)
from .ui import build_demo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emotion-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_predictor()
    yield


app = FastAPI(
    title="Emotion Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s -> %s (%.2f ms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.get("/health", response_model=HealthResponse)
def health():
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model=predictor.info["name"],
        backend=predictor.backend,
        version=predictor.info["version"],
        labels=predictor.labels,
    )


@app.post("/predict", response_model=EmotionPrediction)
def predict(request: PredictRequest):
    result = get_predictor().predict([request.text])[0]
    return EmotionPrediction(**result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    results = get_predictor().predict(request.texts)
    return BatchPredictResponse(predictions=[EmotionPrediction(**r) for r in results])


demo = build_demo()
app = gr.mount_gradio_app(app, demo, path="/ui")
"""

files["artifacts/.gitkeep"] = ""
files["notebooks/.gitkeep"] = ""

for filepath, content in files.items():
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

print("Done! All project folders and files have been created successfully.")
