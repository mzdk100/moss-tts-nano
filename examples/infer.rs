use {
    anyhow::{Error, Result},
    clap::Parser,
    futures_util::StreamExt,
    moss_tts_nano::MossTtsNano,
    std::{path::Path, time::Instant},
    tokio::{spawn, sync::mpsc::channel},
    voxudio::AudioPlayer,
};

const SAMPLE_RATE: usize = 48000;

//noinspection SpellCheckingInspection
#[derive(clap::Parser, Debug)]
#[command(name = "moss-tts-nano", about = "MOSS-TTS-Nano TTS via ONNX Runtime")]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    text: String,

    /// Reference audio path for voice cloning
    #[arg(long)]
    prompt_audio_path: Option<String>,

    /// Built-in voice preset name (default: Junhao)
    #[arg(long, default_value = "Junhao")]
    voice: String,

    /// TTS model directory
    #[arg(long, default_value = "models/MOSS-TTS-Nano-100M-ONNX")]
    model_dir: String,

    /// Codec model directory
    #[arg(long, default_value = "models/MOSS-Audio-Tokenizer-Nano-ONNX")]
    codec_dir: String,

    /// Random seed for reproducibility (default: 1234, matching Python)
    #[arg(long, default_value_t = 1234)]
    seed: u64,

    /// Sample mode: greedy, fixed, or full
    #[arg(long, default_value = "fixed")]
    sample_mode: String,

    /// Use streaming codec decode
    #[arg(long, default_value_t = false)]
    streaming: bool,

    /// Max text tokens per chunk for voice cloning
    #[arg(long, default_value_t = 75)]
    voice_clone_max_text_tokens: usize,

    /// Audio-layer repetition penalty
    #[arg(long, default_value_t = 1.2)]
    audio_repetition_penalty: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Build the TTS engine
    let mut builder = MossTtsNano::builder()
        .model_dir(Path::new(&args.model_dir))
        .codec_dir(Path::new(&args.codec_dir))
        .voice(&args.voice)
        .sample_mode(&args.sample_mode)
        .seed(args.seed)
        .audio_repetition_penalty(args.audio_repetition_penalty)
        .voice_clone_max_text_tokens(args.voice_clone_max_text_tokens);

    if let Some(ref path) = args.prompt_audio_path {
        builder = builder.prompt_audio_path(Path::new(path));
    }

    let mut tts = builder.build().await?;

    println!("All models loaded.");

    // Set up audio playback via voxudio
    let mut player = AudioPlayer::new()?;
    player.play()?;

    if args.streaming {
        // Streaming mode: play audio chunks as they arrive
        let (tx, mut rx) = channel(100);
        let task = spawn(async move {
            let mut stream = tts.synth_stream(&args.text);
            let mut chunk_count = 0;
            let mut all_samples = 0;
            let t_start = Instant::now();

            while let Some(chunk) = stream.next().await {
                chunk_count += 1;
                let chunk = chunk?;
                all_samples += chunk.len();

                println!(
                    "  Received chunk {}: {} samples, at {:?}",
                    chunk_count,
                    chunk.len(),
                    t_start.elapsed()
                );
                tx.send(chunk).await?;
            }

            let synth_elapsed = t_start.elapsed().as_secs_f64();
            let audio_duration = all_samples as f64 / 2.0 / SAMPLE_RATE as f64;
            let rtf = synth_elapsed / audio_duration;
            println!("Synthesis: {:.2}s | RTF: {:.3}", synth_elapsed, rtf,);

            Ok::<_, Error>((audio_duration, chunk_count))
        });
        while let Some(chunk) = rx.recv().await {
            player.write::<SAMPLE_RATE, f32>(&chunk, 2).await?;
        }

        let (audio_duration, chunk_count) = task.await??;
        println!(
            "Playing | Audio: {:.2}s | Chunks: {} | Sample rate: 48000",
            audio_duration, chunk_count
        );
    } else {
        // Non-streaming mode: synthesize all at once, then play
        let t0 = Instant::now();
        let samples = tts.synth(&args.text).await?;
        let synth_elapsed = t0.elapsed().as_secs_f64();

        let audio_duration = samples.len() as f64 / 2.0 / SAMPLE_RATE as f64;
        let rtf = synth_elapsed / audio_duration;

        println!(
            "Playing | Audio: {:.2}s | Sample rate: 48000",
            audio_duration
        );
        println!("Synthesis: {:.2}s | RTF: {:.3}", synth_elapsed, rtf);

        player.write::<48000, f32>(&samples, 2).await?;
    }

    player.stop()?;
    Ok(())
}
