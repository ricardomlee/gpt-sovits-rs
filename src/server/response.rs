//! Response helpers shared by HTTP handlers.

use axum::{
    body::Body,
    http::{header, StatusCode},
    response::Response,
};
use gpt_sovits_rs::Language;

pub(super) fn json_error(status: StatusCode, message: impl AsRef<str>) -> Response<Body> {
    let message = message.as_ref();
    let error_json = serde_json::json!({
        "success": false,
        "error": message,
        "message": message,
    });
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(error_json.to_string()))
        .unwrap()
}

pub(super) fn language_code(language: Language) -> &'static str {
    match language {
        Language::Chinese => "zh",
        Language::English => "en",
        Language::Japanese => "ja",
        Language::Korean => "ko",
        Language::Cantonese => "yue",
        Language::Auto => "auto",
    }
}

fn safe_header_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii() && !ch.is_ascii_control() {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

pub(super) fn add_synthesis_headers(
    mut builder: axum::http::response::Builder,
    voice: Option<&str>,
    language: Language,
    text_chars: usize,
) -> axum::http::response::Builder {
    builder = builder
        .header("x-tts-language", language_code(language))
        .header("x-tts-text-chars", text_chars.to_string());

    if let Some(voice) = voice {
        builder = builder.header("x-tts-voice", safe_header_value(voice));
    }

    builder
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SpeechOutputFormat {
    Wav,
    Pcm,
}

impl SpeechOutputFormat {
    pub(super) fn parse(format: Option<&str>) -> Result<Self, String> {
        let format = format.unwrap_or("wav").trim().to_ascii_lowercase();
        match format.as_str() {
            "wav" => Ok(Self::Wav),
            "pcm" => Ok(Self::Pcm),
            other => Err(format!(
                "unsupported response_format: {other}; supported formats: wav, pcm"
            )),
        }
    }

    pub(super) fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Pcm => "application/octet-stream",
        }
    }

    pub(super) fn as_header_value(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_non_ascii_header_values() {
        assert_eq!(safe_header_value("mao"), "mao");
        assert_eq!(safe_header_value("角色 A"), "__ A");
    }

    #[test]
    fn accepts_only_lossless_speech_formats_for_openai_endpoint() {
        assert_eq!(
            SpeechOutputFormat::parse(None).unwrap(),
            SpeechOutputFormat::Wav
        );
        assert_eq!(
            SpeechOutputFormat::parse(Some("wav")).unwrap(),
            SpeechOutputFormat::Wav
        );
        assert_eq!(
            SpeechOutputFormat::parse(Some(" PCM ")).unwrap(),
            SpeechOutputFormat::Pcm
        );
        assert!(SpeechOutputFormat::parse(Some("mp3")).is_err());
        assert!(SpeechOutputFormat::parse(Some("opus")).is_err());
    }
}
