use super::*;
use std::io::Cursor;
use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error, formats::FormatOptions,
    io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};

fn tone(len: usize, rate: u32) -> AudioBuffer {
    let samples = (0..len)
        .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / rate as f32).sin() * 0.2)
        .collect();
    AudioBuffer::new(samples, rate, 1)
}

fn decode_mp3(bytes: Vec<u8>) -> (Vec<f32>, u32) {
    let stream = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension("mp3");
    let mut reader = symphonia::default::get_probe()
        .format(
            &hint,
            stream,
            &FormatOptions {
                enable_gapless: true,
                ..Default::default()
            },
            &MetadataOptions::default(),
        )
        .unwrap()
        .format;
    let track = reader.default_track().unwrap();
    let rate = track.codec_params.sample_rate.unwrap();
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .unwrap();
    let mut samples = Vec::new();
    loop {
        let packet = match reader.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => panic!("MP3 packet: {e}"),
        };
        let audio = decoder.decode(&packet).unwrap();
        assert_eq!(audio.spec().channels.count(), 1);
        let mut buffer = SampleBuffer::<f32>::new(audio.capacity() as u64, *audio.spec());
        buffer.copy_interleaved_ref(audio);
        samples.extend_from_slice(buffer.samples());
    }
    (samples, rate)
}

fn decode_opus(bytes: Vec<u8>) -> Vec<f32> {
    let mut reader = ogg::reading::PacketReader::new(Cursor::new(bytes));
    let head = reader.read_packet_expected().unwrap();
    assert!(head.first_in_stream());
    assert!(head.last_in_page());
    assert_eq!(&head.data[..10], b"OpusHead\x01\x01");
    assert_eq!(head.data.len(), 19);
    let pre_skip = u16::from_le_bytes(head.data[10..12].try_into().unwrap()) as usize;
    let tags = reader.read_packet_expected().unwrap();
    assert!(tags.last_in_page());
    assert!(tags.data.starts_with(b"OpusTags"));
    let mut decoder = ::opus::Decoder::new(48_000, ::opus::Channels::Mono).unwrap();
    let mut pcm = [0.0; 5760];
    let mut samples = Vec::new();
    let mut last_granule = None;
    while let Some(packet) = reader.read_packet().unwrap() {
        let count = decoder.decode_float(&packet.data, &mut pcm, false).unwrap();
        samples.extend_from_slice(&pcm[..count]);
        if packet.last_in_stream() {
            last_granule = Some(packet.absgp_page() as usize);
        }
    }
    let end = last_granule.expect("Opus stream must finish with EOS");
    assert!(end <= samples.len());
    assert!(pre_skip < end);
    samples[pre_skip..end].to_vec()
}

fn assert_signal(samples: &[f32]) {
    assert!(samples.iter().all(|v| v.is_finite()));
    for part in [samples, &samples[..2400], &samples[samples.len() - 2400..]] {
        let rms = (part.iter().map(|v| v * v).sum::<f32>() / part.len() as f32).sqrt();
        assert!((0.08..0.20).contains(&rms), "invalid RMS: {rms}");
    }
}

#[test]
fn mp3_roundtrip_preserves_rate_duration_and_signal() {
    for rate in [24_000, 32_000, 44_100, 48_000] {
        let source = tone(rate as usize + 123, rate);
        let encoded = encode(&source, SpeechOutputFormat::Mp3).unwrap();
        assert_eq!(encoded.sample_rate, rate);
        assert!(encoded.bytes.len() < source.samples.len() * 2);
        let (samples, decoded_rate) = decode_mp3(encoded.bytes);
        assert_eq!(decoded_rate, rate);
        assert_eq!(
            samples.len(),
            source.samples.len(),
            "gapless MP3 length at {rate}"
        );
        assert_signal(&samples);
    }
}

#[test]
fn opus_roundtrip_preserves_duration_and_signal() {
    for rate in [24_000, 32_000, 44_100, 48_000] {
        let source = tone(rate as usize + 123, rate);
        let encoded = encode(&source, SpeechOutputFormat::Opus).unwrap();
        assert_eq!(encoded.sample_rate, 48_000);
        assert!(encoded.bytes.starts_with(b"OggS"));
        assert!(encoded.bytes.len() < source.samples.len() * 2);
        let samples = decode_opus(encoded.bytes);
        let expected = (source.samples.len() as f64 * 48_000.0 / rate as f64).round() as usize;
        assert_eq!(samples.len(), expected);
        assert_signal(&samples);
    }
}

#[test]
fn opus_spanning_multiple_ogg_pages_preserves_the_tail() {
    let source = tone(32_000 * 7 + 71, 32_000);
    let encoded = encode(&source, SpeechOutputFormat::Opus).unwrap();
    let decoded = decode_opus(encoded.bytes);
    assert_eq!(
        decoded.len(),
        (source.samples.len() as f64 * 1.5).round() as usize
    );
    assert_signal(&decoded);
}

#[test]
fn codecs_flush_short_and_frame_boundary_audio() {
    for len in [1, 10, 639, 640, 641, 959, 960, 961, 1151, 1152, 1153] {
        let source = AudioBuffer::new(vec![0.1; len], 48_000, 1);
        let mp3 = encode(&source, SpeechOutputFormat::Mp3).unwrap();
        assert_eq!(decode_mp3(mp3.bytes).0.len(), len, "MP3 {len}");
        let opus = encode(&source, SpeechOutputFormat::Opus).unwrap();
        assert_eq!(decode_opus(opus.bytes).len(), len, "Opus {len}");
    }
}

#[test]
fn wav_and_pcm_bytes_remain_unchanged() {
    let source = tone(3200, 32_000);
    assert_eq!(
        encode(&source, SpeechOutputFormat::Wav).unwrap().bytes,
        source.to_wav_bytes().unwrap()
    );
    assert_eq!(
        encode(&source, SpeechOutputFormat::Pcm).unwrap().bytes,
        samples_to_pcm(&source.samples)
    );
}

#[test]
fn malformed_audio_is_rejected_before_calling_codecs() {
    for audio in [
        AudioBuffer::empty(32_000, 1),
        AudioBuffer::new(vec![0.0], 0, 1),
        AudioBuffer::new(vec![0.0, 0.0], 32_000, 2),
        AudioBuffer::new(vec![f32::NAN], 32_000, 1),
        AudioBuffer::new(vec![f32::INFINITY], 32_000, 1),
    ] {
        for format in [
            SpeechOutputFormat::Wav,
            SpeechOutputFormat::Pcm,
            SpeechOutputFormat::Mp3,
            SpeechOutputFormat::Opus,
        ] {
            assert!(encode(&audio, format).is_err());
        }
    }
}
