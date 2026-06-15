# MOSS-TTS-Nano (Rust / ONNX Runtime)

[English](README.md) | 中文

基于 ONNX Runtime 的 [MOSS-TTS-Nano](https://github.com/OpenMOSS/moss-tts) 高性能 Rust 推理实现。具备真正的流式合成能力，首包延迟低至 ~100ms，支持声音克隆和多种采样策略。

## 特性

- **真流式合成** — 帧生成与音频编解码交错执行；每生成一帧立即解码并输出音频，无需等待整段语音生成完毕。
- **零拷贝张量传递** — ONNX 输入使用 `TensorRef` 引用传递，避免每步推理时克隆张量。
- **声音克隆** — 提供参考 WAV 文件即可克隆任意音色。
- **内置音色预设** — 自带 `Junhao` 音色，可通过 JSON 元数据扩展。
- **三种采样模式** — `greedy`（贪心）、`fixed`（单次 ONNX 调用）、`full`（逐通道自回归 + 本地 KV 缓存）。
- **自适应流式解码** — 编解码帧预算根据已解码音频的领先时间在 1~8 帧之间动态调整。
- **纯 Rust SentencePiece** — 无需 C++ 依赖，分词器完全用 Rust 实现。
- **全异步设计** — 所有 ONNX 推理通过 tokio 的 `run_async()` 执行，天然非阻塞。
- **48kHz 立体声输出** — 高保真 48kHz 采率音频。

## 环境要求

- Rust 1.85+（edition 2024）
- ONNX Runtime（通过 `ort` crate 自动捆绑）

下载 ONNX 模型文件并放置到 `models/` 目录：

```bash
# 使用 huggingface-cli 下载
huggingface-cli download OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX --local-dir models/MOSS-TTS-Nano-100M-ONNX
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX --local-dir models/MOSS-Audio-Tokenizer-Nano-ONNX
```

或从 Hugging Face 手动下载：
- [MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- [MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

## 获取

```bash
cargo add moss-tts-nano
```

## 构建

```bash
git clone https://github.com/mzdk100/moss-tts-nano
cd moss-tts-nano
cargo build --release
```

> 必须使用 release 模式，否则 ONNX 推理性能无法接受。

## 运行

```bash
# 使用内置音色合成
cargo run --release --example infer -- --text "Hello, world." --streaming

# 使用参考音频进行声音克隆
cargo run --release --example infer -- --text "Hello, world." --prompt-audio-path assets/hello_in_cn.wav --streaming

# 中文文本
cargo run --release --example infer -- --text "今天天气不错，适合出去走走" --prompt-audio-path assets/test.wav --streaming

# 不同采样模式
cargo run --release --example infer -- --text "Hello" --sample-mode greedy
cargo run --release --example infer -- --text "Hello" --sample-mode fixed
cargo run --release --example infer -- --text "Hello" --sample-mode full
```

## 库使用示例

```rust
use moss_tts_nano::{MossTtsNano, TtsError};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<(), TtsError> {
    // 构建引擎
    let mut tts = MossTtsNano::builder()
        .model_dir("models/MOSS-TTS-Nano-100M-ONNX")
        .codec_dir("models/MOSS-Audio-Tokenizer-Nano-ONNX")
        .voice("Junhao")                // 或 .prompt_audio_path("ref.wav")
        .sample_mode("fixed")           // greedy | fixed | full
        .seed(1234)
        .build().await?;

    // 批量合成
    let samples: Vec<f32> = tts.synth("你好，世界。").await?;

    // 流式合成 — 音频块随解码即时到达（首包 ~100ms）
    let mut stream = tts.synth_stream("你好，世界。");
    while let Some(chunk) = stream.next().await {
        let pcm: Vec<f32> = chunk?;
        // 播放或写入 pcm...
    }

    Ok(())
}
```

## 架构

```
文本 → 分词器 → 预填充 (ONNX) → 自回归帧循环 → 编解码 → 48kHz 音频
                                    ↓ 逐帧
                              生成帧 token
                                    ↓
                              编解码（交错执行）
                                    ↓
                              输出音频块
```

流水线包含三个阶段：

1. **预填充** — 文本经分词后与提示/参考音频 token 拼接，一次 ONNX prefill 调用产生初始隐藏状态和 KV 缓存。
2. **帧生成** — 自回归循环逐帧生成音频 token。每帧由 `n_vq` 个码本索引组成。通过帧重复分析检测语音结束。
3. **编解码** — 编解码模型将帧解码为 PCM 音频。流式模式下，编解码与帧生成交错执行——每生成一帧立即解码。

启动时加载七个 ONNX 会话：`prefill`、`decode_step`、`local_fixed_sampled_frame`、`local_cached_step`、`codec_encode`、`codec_decode`、`codec_decode_step`。

## 模型来源

ONNX 模型由 [moss-tts](https://github.com/OpenMOSS/moss-tts) 项目从官方 PyTorch 权重转换而来：

- **TTS Transformer**: [OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)
- **Audio Codec**: [OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX)

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可。
