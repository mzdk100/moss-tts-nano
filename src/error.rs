//noinspection SpellCheckingInspection
use {
    ndarray::ShapeError,
    ort::Error as OrtError,
    serde_json::Error as SerdeJsonError,
    std::{
        error::Error,
        fmt::{Display, Formatter, Result as FmtResult},
        io::Error as IoError,
        str::Utf8Error,
        string::FromUtf8Error,
    },
    voxudio::OperationError as VoxudioError,
};

//noinspection SpellCheckingInspection
/// Error type for MOSS-TTS-Nano operations.
#[derive(Debug)]
pub enum TtsError {
    /// UTF-8 decoding error.
    FromUtf8(FromUtf8Error),
    /// ONNX Runtime error.
    Ort(String),
    /// I/O error (file read/write).
    Io(IoError),
    /// Serde JSON error.
    SerdeJson(SerdeJsonError),
    /// NDArray error.
    Shape(ShapeError),
    /// Voxudio error.
    Voxudio(VoxudioError),
    /// UTF-8 error.
    Utf8(Utf8Error),
    /// Configuration or input error.
    Config(String),
    /// Varint decoding error.
    EndOfVarint(String),
    /// Invalid wire type.
    InvalidWireType(String),
    /// Model produced no audio frames.
    NoAudioGenerated,
    /// Varint too long.
    VarintTooLong(String),
}

impl Display for TtsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "TtsError: ")?;

        match self {
            Self::FromUtf8(e) => Display::fmt(e, f),
            Self::Io(e) => Display::fmt(e, f),
            Self::Ort(msg) => write!(f, "OrtError: {}", msg),
            Self::SerdeJson(e) => Display::fmt(e, f),
            Self::Shape(e) => Display::fmt(e, f),
            Self::Voxudio(e) => Display::fmt(e, f),
            Self::Utf8(e) => Display::fmt(e, f),
            Self::Config(msg) => write!(f, "ConfigError: {}", msg),
            Self::EndOfVarint(msg) => write!(f, "EndOfVarintError: {}", msg),
            Self::InvalidWireType(msg) => write!(f, "InvalidWireTypeError: {}", msg),
            Self::NoAudioGenerated => write!(f, "NoAudioGeneratedError: no audio frames generated"),
            Self::VarintTooLong(msg) => write!(f, "VarintTooLongError: {}", msg),
        }
    }
}

impl Error for TtsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TtsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl<T> From<OrtError<T>> for TtsError {
    fn from(e: OrtError<T>) -> Self {
        Self::Ort(e.to_string())
    }
}

impl From<IoError> for TtsError {
    fn from(e: IoError) -> Self {
        Self::Io(e)
    }
}

impl From<SerdeJsonError> for TtsError {
    fn from(e: SerdeJsonError) -> Self {
        Self::SerdeJson(e)
    }
}

impl From<ShapeError> for TtsError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

impl From<FromUtf8Error> for TtsError {
    fn from(e: FromUtf8Error) -> Self {
        Self::FromUtf8(e)
    }
}

impl From<VoxudioError> for TtsError {
    fn from(e: VoxudioError) -> Self {
        Self::Voxudio(e)
    }
}

impl From<Utf8Error> for TtsError {
    fn from(e: Utf8Error) -> Self {
        Self::Utf8(e)
    }
}
