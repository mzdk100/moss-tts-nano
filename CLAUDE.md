# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Build library (release mode required for acceptable ONNX performance)
cargo build --release

# Run inference example
cargo run --release --example infer -- --text "Hello, world."

# Voice cloning from reference audio
cargo run --release --example infer -- --text "Hello" --prompt-audio-path assets/hello_in_cn.wav

# Built-in voice preset
cargo run --release --example infer -- --text "Hello" --voice Junhao

# Sampling modes: greedy | fixed | full
cargo run --release --example infer -- --text "Hello" --sample-mode greedy

# Streaming decode mode
cargo run --release --example infer -- --text "Hello" --streaming

# Quality checks
cargo clippy --release -- -W clippy::all
cargo fmt --check
cargo test
```

## Prerequisites

ONNX model files must be placed in `models/` (gitignored). Two model directories required:
- `models/MOSS-TTS-Nano-100M-ONNX/` — TTS transformer model
- `models/MOSS-Audio-Tokenizer-Nano-ONNX/` — Audio codec model

## Public API

The library exposes a fully async API via `MossTtsNano`. All ONNX inference uses `run_async()` internally. Requires a tokio runtime.

```rust
use moss_tts_nano::{MossTtsNano, TtsError};

// Build the engine (async — loads ONNX models)
let mut tts = MossTtsNano::builder()
    .model_dir("models/MOSS-TTS-Nano-100M-ONNX")
    .codec_dir("models/MOSS-Audio-Tokenizer-Nano-ONNX")
    .voice("Junhao")           // or .prompt_audio_path("ref.wav")
    .sample_mode("fixed")      // greedy | fixed | full
    .seed(1234)
    .build().await?;

// Batch synthesis → interleaved f32 samples
let samples: Vec<f32> = tts.synth("Hello, world.").await?;

// Streaming synthesis → chunks arrive as decoded
let mut stream = tts.synth_stream("Hello, world.");
while let Some(chunk) = stream.next().await {
    let samples: Vec<f32> = chunk?;
}
```

Key types: `MossTtsNano`, `MossTtsNanoBuilder`, `TtsError`.

## Architecture

Rust library + CLI for neural TTS inference via ONNX Runtime. ~100M parameter model with two-stage architecture: global transformer (12 layers, 768 hidden) generates audio token frames, local decoder refines them, codec decodes tokens to 48kHz audio.

### Module Layout (`src/`)

All modules are private; `lib.rs` exposes only `MossTtsNano`, `MossTtsNanoBuilder`, and `TtsError`.

| Module | Role |
|--------|------|
| `lib.rs` | `MossTtsNano` facade, `synth()`/`synth_stream()`. The streaming path inlines the full generation loop with interleaved codec decode. |
| `builder.rs` | `MossTtsNanoBuilder` — builder pattern for constructing `MossTtsNano`. Loads models, resolves voice presets, encodes reference audio. |
| `error.rs` | `TtsError` enum (Ort, Io, Config, NoAudioGenerated) |
| `config.rs` | `AppConfig`, `TtsModelConfig`, `CodecConfig`, `GenerationConfig`, `SampleMode`, prompt templates, built-in voices. Loads from JSON metadata files in model dirs. |
| `models.rs` | ONNX `Sessions` (7 sessions: prefill, decode_step, local_fixed, local_cached, codec_encode, codec_decode, codec_decode_step). All `run_*` functions use `TensorRef::from_array_view` for zero-copy input passing. |
| `generation.rs` | Per-frame generation helpers (`generate_frame_fixed`, `generate_frame_full`), `EndOfSpeechDetector`, `build_next_row`, numpy-compatible PCG64 RNG. |
| `sampling.rs` | Token sampling: temperature, top-k, top-p, repetition penalty. |
| `codec.rs` | `StreamingDecodeSession` for real-time codec decode with KV cache. Adaptive frame budget (1/2/4/8 frames) based on playback lead time. |
| `tokenizer.rs` | CJK-aware text normalization, sentence/clause splitting, token-budget chunking for voice cloning. |
| `sp_model.rs` | **Private.** Pure Rust SentencePiece (manual protobuf decode, Unigram/BPE). No C++ dependency. |
| `audio.rs` | WAV load/save, linear-interpolation resampling, waveform concatenation, pause generation. |

### Key Design Decisions

- **True streaming**: `synth_stream` inlines the autoregressive generation loop, interleaving frame generation with codec decode. Each frame is decoded to audio immediately after generation, yielding chunks to the consumer. First-packet latency is ~1 frame time (~100ms), not full-utterance time.
- **TensorRef for zero-copy**: All ONNX input tensors use `TensorRef::from_array_view(&array)` instead of `Tensor::from_array(array.clone())`. State tensors (KV caches, attention caches) are passed by reference, eliminating deep clones on every decode step.
- **Fully async**: All ONNX inference uses `run_async()` via `RunOptions`. Requires tokio runtime.
- **`async_stream::stream!`**: `synth_stream` uses the `stream!` macro to safely borrow `&mut self` without unsafe code. Sessions are destructured to avoid borrow conflicts.
- **Pure Rust SentencePiece**: `sp_model.rs` manually decodes protobuf and runs Viterbi/BPE — avoids C++ sentencepiece dependency
- **Numpy-compatible RNG**: PCG64 with pre-computed states for seeds 1234/42 to match Python behavior exactly
- **Three sampling modes**: `greedy` (argmax), `fixed` (single ONNX call per frame), `full` (per-channel autoregressive with local KV cache)
- **Adaptive codec decode**: Frame budget scales from 1 to 8 based on decoded audio lead time vs real-time playback
- **End-of-speech detection**: `EndOfSpeechDetector` tracks recent frames; stops on 8 consecutive identical frames, dominant token >75%, or 2 unique tokens cycling
