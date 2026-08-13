"""
Neuro Fusion-RAG — Speech & Text Branch Demo (Streamlit version)
------------------------------------------------------------------
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

import streamlit as st
import torch
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2Processor, Wav2Vec2Model,
)
import librosa

from src.model import UnimodalClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "Control (no dementia signal)", 1: "Dementia signal detected"}


# ---------------------------------------------------------------------------
# Cached loaders — run once, then reused across every user interaction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_norm_stats():
    return torch.load("preprocessing_config/norm_stats.pt", map_location="cpu", weights_only=False)


@st.cache_resource
def load_text_model():
    model = UnimodalClassifier(input_dim=768).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model_text.pth", map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_speech_model():
    model = UnimodalClassifier(input_dim=768).to(DEVICE)
    model.load_state_dict(torch.load("models/best_model_speech.pth", map_location=DEVICE))
    model.eval()
    return model


@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_wav2vec():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)
    model.eval()
    return processor, model


def normalize(x, mean_key, std_key, norm_stats):
    mean = norm_stats[mean_key]
    std = norm_stats[std_key]
    return (x - mean) / std


def predict_text(transcript: str):
    norm_stats = load_norm_stats()
    text_model = load_text_model()
    bert_tokenizer, bert_model = load_bert()

    with torch.no_grad():
        tokens = bert_tokenizer(
            transcript, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(DEVICE)
        hidden = bert_model(**tokens).last_hidden_state  # [1, seq_len, 768]
        embedding = hidden.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]

        embedding = normalize(embedding, "text_mean", "text_std", norm_stats).unsqueeze(0).to(DEVICE)
        logit = text_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > 0.5)
    return LABELS[pred], round(prob, 4)


def predict_speech(audio_path: str):
    norm_stats = load_norm_stats()
    speech_model = load_speech_model()
    wav2vec_processor, wav2vec_model = load_wav2vec()

    with torch.no_grad():
        speech_array, _ = librosa.load(audio_path, sr=16000)
        inputs = wav2vec_processor(speech_array, sampling_rate=16000, return_tensors="pt").to(DEVICE)
        hidden = wav2vec_model(**inputs).last_hidden_state  # [1, T, 768]
        embedding = hidden.mean(dim=1).squeeze(0).cpu()  # mean-pooled -> [768]

        embedding = normalize(embedding, "speech_mean", "speech_std", norm_stats).unsqueeze(0).to(DEVICE)
        logit = speech_model(embedding)
        prob = torch.sigmoid(logit).item()

    pred = int(prob > 0.5)
    return LABELS[pred], round(prob, 4)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Neuro Fusion-RAG — Speech & Text Branch", layout="centered")

st.title("Neuro Fusion-RAG — Speech & Text Branch Demo")
st.markdown(
    "Text-only and speech-only ablation models from the Pitt Corpus study. "
    "Fusion (MRI + acoustic) branch not included in this demo — see model card below."
)

tab_text, tab_speech = st.tabs(["Text", "Speech"])

with tab_text:
    transcript_in = st.text_area(
        "Transcript",
        placeholder="Paste a Cookie Theft description or other speech transcript...",
        height=160,
    )
    if st.button("Analyze Text"):
        if not transcript_in or not transcript_in.strip():
            st.warning("Please enter a transcript.")
        else:
            with st.spinner("Running BERT + classifier..."):
                label, prob = predict_text(transcript_in)
            st.success(f"Prediction: {label}")
            st.metric("Dementia probability (0-1)", prob)

with tab_speech:
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if audio_file is not None:
        st.audio(audio_file)
    if st.button("Analyze Audio"):
        if audio_file is None:
            st.warning("Please upload audio.")
        else:
            # Save uploaded file to a temp path so librosa can read it
            temp_path = f"/tmp/{audio_file.name}"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            with st.spinner("Running Wav2Vec2 + classifier..."):
                label, prob = predict_speech(temp_path)
            st.success(f"Prediction: {label}")
            st.metric("Dementia probability (0-1)", prob)

st.markdown(
    "### Model card / limitations\n"
    "- Trained on the Pitt Corpus (Cookie Theft, Verbal Fluency, Sentence Construction tasks), "
    "929 samples, evaluated with 4-fold CV + external ADReSS/Addresso validation.\n"
    "- Reported test accuracy: **92.14%** (text-only), **80.71%** (speech-only).\n"
    "- This demo builds embeddings on the fly with `bert-base-uncased` (mean-pooled) and "
    "`facebook/wav2vec2-base-960h` (mean-pooled). If the original training embeddings used a "
    "different pooling strategy or model checkpoint, live predictions may not match reported accuracy.\n"
    "- **Not a medical device.** Research demo only — not for clinical use."
)
