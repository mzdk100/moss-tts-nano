# MOSS-TTS-Nano (Rust / ONNX Runtime)

English | [中文](README_CN.md)

A high-performance Rust implementation of [MOSS-TTS-Nano](https://github.com/OpenMOSS/moss-tts) inference via ONNX Runtime. Features true streaming synthesis with minimal first-packet latency (~100ms), voice cloning from reference audio, and multiple sampling strategies.

## Features

- **True streaming** — Frame generation and codec decode are interleaved; audio chunks are yielded as each frame is generated, not after the entire utterance.
- **Zero-copy tensor passing** — ONNX inputs use `TensorRef` borrows instead of cloning tensors on every inference step.
- **Voice cloning** — Provide a reference WAV file to clone any voice.
- **Built-in voice presets** — Ships with the `Junhao` voice preset; extensible via JSON metadata.
- **Three sampling modes** — `greedy` (argmax), `fixed` (single ONNX call per frame), `full` (per-channel autoregressive with local KV cache).
- **Adaptive streaming** — Codec decode frame budget scales from 1 to 8 based on decoded audio lead time vs real-time playback.
- **Pure Rust SentencePiece** — No C++ dependency; SentencePiece tokenizer is implemented entirely in Rust.
- **Fully async** — All ONNX inference uses `run_async()` via tokio. Non-blocking by design.
- **48kHz stereo output** — High-fidelity audio at 48kHz sample rate.

## Prerequisites

- Rust 1.85+ (edition 2024)
- ONNX Runtime (bundled via the `ort` crate)

Download the ONNX model files and place them in the `models/` directory:

```bash
# Using huggingface-cli
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir models/MOSS-TTS-Nano-100M-ONNX
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir models/MOSS-Audio-Tokenizer-Nano-ONNX
```

Or download manually from Hugging Face:
- [MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- [MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

## Installation

```bash
cargo add moss-tts-nano
```

## Build

```bash
git clone https://github.com/mzdk100/moss-tts-nano
cd moss-tts-nano
cargo build --release
```

> Release mode is required for acceptable ONNX inference performance.

## Run

```bash
# Basic synthesis with built-in voice
cargo run --release --example infer -- --text "Hello, world." --streaming

# Voice cloning from reference audio
cargo run --release --example infer -- --text "Hello, world." --prompt-audio-path assets/hello_in_cn.wav --streaming

# Chinese text
cargo run --release --example infer -- --text "今天天气不错，适合出去走走" --prompt-audio-path assets/test.wav --streaming

# Sampling modes
cargo run --release --example infer -- --text "Hello" --sample-mode greedy
cargo run --release --example infer -- --text "Hello" --sample-mode fixed
cargo run --release --example infer -- --text "Hello" --sample-mode full
```

## Library Usage

```rust
use moss_tts_nano::{MossTtsNano, TtsError};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), TtsError> {
    // Build the engine
    let mut tts = MossTtsNano::builder()
        .model_dir("models/MOSS-TTS-Nano-100M-ONNX")
        .codec_dir("models/MOSS-Audio-Tokenizer-Nano-ONNX")
        .voice("Junhao")                // or .prompt_audio_path("ref.wav")
        .sample_mode("fixed")           // greedy | fixed | full
        .seed(1234)
        .build().await?;

    // Batch synthesis
    let samples: Vec<f32> = tts.synth("Hello, world.").await?;

    // Streaming synthesis — chunks arrive as decoded (~100ms first packet)
    let mut stream = tts.synth_stream("Hello, world.");
    while let Some(chunk) = stream.next().await {
        let pcm: Vec<f32> = chunk?;
        // play or write pcm...
    }

    Ok(())
}
```

## Architecture

```
text → tokenizer → prefill (ONNX) → autoregressive frame loop → codec decode → 48kHz audio
                                        ↓ per frame
                                  generate frame tokens
                                        ↓
                                  codec decode (interleaved)
                                        ↓
                                  yield audio chunk
```

The pipeline has three stages:

1. **Prefill** — Text is tokenized and combined with prompt/reference audio tokens. A single ONNX prefill call produces the initial hidden states and KV cache.
2. **Frame generation** — An autoregressive loop generates audio token frames one at a time. Each frame is a vector of `n_vq` codebook indices. End-of-speech is detected via frame repetition analysis.
3. **Codec decode** — Frames are decoded to PCM audio by the codec model. In streaming mode, this is interleaved with frame generation — each frame is decoded immediately after it is generated.

Seven ONNX sessions are loaded at startup: `prefill`, `decode_step`, `local_fixed_sampled_frame`, `local_cached_step`, `codec_encode`, `codec_decode`, `codec_decode_step`.

## Model Sources

The ONNX models are converted from the official PyTorch weights by the [moss-tts](https://github.com/OpenMOSS/moss-tts) project:

- **TTS Transformer**: [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- **Audio Codec**: [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
