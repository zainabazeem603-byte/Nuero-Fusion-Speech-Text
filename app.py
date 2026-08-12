"""
Neuro Fusion-RAG — Speech & Text Branch Demo
----------------------------------------------
Live demo for the text-only and speech-only ablation models trained on the
Pitt Corpus (Cookie Theft / Verbal Fluency / Sentence Construction tasks).

NOTE: The fusion model (text + speech + acoustic) is NOT included here because
the original 47-dim acoustic feature extractor used during training could not
be confirmed. Only the two unimodal branches — which take a single 768-dim
embedding each — are deployed:
  - Text branch  : BERT (bert-base-uncased), mean-pooled last hidden state
  - Speech branch: Wav2Vec2 (facebook/wav2vec2-base-960h), mean-pooled over time

Because the exact pooling/tokenization used to build the original training
embeddings isn't recorded in the project files either, predictions here are
best-effort — accuracy may differ from the reported test-set numbers
(92.14% text-only, 80.71% speech-only) if the original pipeline pooled
differently (e.g. CLS token instead of mean pooling).
"""

import torch
import torch.nn.functional as F
import gradio as gr
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2Processor, Wav2Vec2Model,
)
import librosa
import numpy as np

from src.model import UnimodalClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load normalization stats (computed on training split only)
# ---------------------------------------------------------------------------
norm_stats = torch.load("preprocessing_config/norm_stats.pt", map_location="cpu", weights_only=False)


def normalize(x, mean_key, std_key):
    mean = norm_stats[mean_key]
    std = norm_stats[std_key]
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Load classifiers
# ---------------------------------------------------------------------------
text_model = UnimodalClassifier(input_dim=768).to(DEVICE)
text_model.load_state_dict(torch.load("models/best_model_text.pth", map_location=DEVICE))
text_model.eval()

speech_model = UnimodalClassifier(input_dim=768).to(DEVICE)
speech_model.load_state_dict(torch.load("models/best_model_speech.pth", map_location=DEVICE))
speech_model.eval()

# ---------------------------------------------------------------------------
# Load embedding backbones (lazy globals, loaded once at startup)
# ---------------------------------------------------------------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
wav2vec_model.eval()


LABELS = {0: "Control (no dementia signal)", 1: "Dementia signal detected"}


def predict_text(transcript: str):
    if not transcript or not transcript.strip():
        return "Please enter a transcript.", None

    with torch.no_grad():
        tokens = bert_tokenizer(
            transcript, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        hidden = bert_model(**tokens).last_hidden_state  # [1, seq_len, 768]
        embedding = hidden.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]

        embedding = normalize(embedding, "text_mean", "text_std").unsqueeze(0).to(DEVICE)
        logit = text_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > 0.5)
    return LABELS[pred], round(prob, 4)


def predict_speech(audio_path: str):
    if audio_path is None:
        return "Please upload or record audio.", None

    with torch.no_grad():
        speech_array, _ = librosa.load(audio_path, sr=16000)
        inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        hidden = wav2vec_model(**inputs).last_hidden_state  # [1, T, 768]
        embedding = hidden.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]

        embedding = normalize(embedding, "speech_mean", "speech_std").unsqueeze(0).to(DEVICE)
        logit = speech_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > 0.5)
    return LABELS[pred], round(prob, 4)


with gr.Blocks(title="Neuro Fusion-RAG — Speech & Text Branch") as demo:
    gr.Markdown(
        "# Neuro Fusion-RAG — Speech & Text Branch Demo\n"
        "Text-only and speech-only ablation models from the Pitt Corpus study. "
        "Fusion (MRI + acoustic) branch not included in this demo — see model card below."
    )

    with gr.Tab("Text"):
        transcript_in = gr.Textbox(
            label="Transcript",
            placeholder="Paste a Cookie Theft description or other speech transcript...",
            lines=6,
        )
        text_btn = gr.Button("Analyze Text")
        text_label = gr.Textbox(label="Prediction")
        text_prob = gr.Number(label="Dementia probability (0-1)")
        text_btn.click(predict_text, inputs=transcript_in, outputs=[text_label, text_prob])

    with gr.Tab("Speech"):
        audio_in = gr.Audio(label="Upload or record audio", type="filepath")
        speech_btn = gr.Button("Analyze Audio")
        speech_label = gr.Textbox(label="Prediction")
        speech_prob = gr.Number(label="Dementia probability (0-1)")
        speech_btn.click(predict_speech, inputs=audio_in, outputs=[speech_label, speech_prob])

    gr.Markdown(
        "### Model card / limitations\n"
        "- Trained on the Pitt Corpus (Cookie Theft, Verbal Fluency, Sentence Construction tasks), "
        "929 samples, evaluated with 4-fold CV + external ADReSS/Addresso validation.\n"
        "- Reported test accuracy: **92.14%** (text-only), **80.71%** (speech-only).\n"
        "- This demo builds embeddings on the fly with `bert-base-uncased` (mean-pooled) and "
        "`facebook/wav2vec2-base-960h` (mean-pooled). If the original training embeddings used a "
        "different pooling strategy or model checkpoint, live predictions may not match reported accuracy.\n"
        "- **Not a medical device.** Research demo only — not for clinical use."
    )

demo.launch()
