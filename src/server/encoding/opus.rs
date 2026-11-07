use anyhow::{anyhow, ensure, Context, Result};
use gpt_sovits_rs::AudioBuffer;
use ogg::writing::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Encoder};
use soxr::{format::Mono, Soxr};

const RATE: u32 = 48_000;
const FRAME: usize = 960;

fn resample(audio: &AudioBuffer) -> Result<Vec<f32>> {
    if audio.sample_rate == RATE {
        return Ok(audio.samples.clone());
    }
    let size = (audio.samples.len() as u64 * u64::from(RATE)).div_ceil(u64::from(audio.sample_rate))
        as usize
        + 64;
    let mut output = vec![0.0; size];
    let mut resampler = Soxr::<Mono<f32>>::new(audio.sample_rate as f64, RATE as f64)
        .map_err(|error| anyhow!("soxr init: {error}"))?;
    let processed = resampler
        .process(&audio.samples, &mut output)
        .map_err(|error| anyhow!("soxr process: {error}"))?;
    ensure!(
        processed.input_frames == audio.samples.len(),
        "incomplete Opus resampling"
    );
    output.truncate(processed.output_frames);
    let mut tail = [0.0; 4096];
    loop {
        let count = resampler
            .drain(&mut tail)
            .map_err(|error| anyhow!("soxr drain: {error}"))?;
        output.extend_from_slice(&tail[..count]);
        if count == 0 {
            break;
        }
    }
    Ok(output)
}

pub(super) fn encode(audio: &AudioBuffer) -> Result<Vec<u8>> {
    let samples = resample(audio).context("Opus resampling failed")?;
    let mut encoder = Encoder::new(RATE, Channels::Mono, Application::Audio)?;
    encoder.set_bitrate(Bitrate::Bits(48_000))?;
    let pre_skip = u16::try_from(encoder.get_lookahead()?)?;
    let mut writer = PacketWriter::new(Vec::new());
    let serial = rand::random();

    // RFC 7845: headers occupy separate pages; granules always count 48 kHz samples.
    let mut head = b"OpusHead".to_vec();
    head.extend_from_slice(&[1, 1]);
    head.extend_from_slice(&pre_skip.to_le_bytes());
    head.extend_from_slice(&audio.sample_rate.to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes());
    head.push(0);
    writer.write_packet(head, serial, PacketWriteEndInfo::EndPage, 0)?;

    let vendor = b"gpt-sovits-rs";
    let mut tags = b"OpusTags".to_vec();
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes());
    writer.write_packet(tags, serial, PacketWriteEndInfo::EndPage, 0)?;

    // Flush encoder lookahead with silence, then trim only the padding via EOS granule.
    let total = samples.len() + usize::from(pre_skip);
    let frames = total.div_ceil(FRAME);
    let mut pcm = [0.0; FRAME];
    let mut packet = [0u8; 4000];
    for index in 0..frames {
        pcm.fill(0.0);
        let start = index * FRAME;
        if start < samples.len() {
            let count = FRAME.min(samples.len() - start);
            for (target, sample) in pcm[..count].iter_mut().zip(&samples[start..start + count]) {
                *target = sample.clamp(-1.0, 1.0);
            }
        }
        let len = encoder.encode_float(&pcm, &mut packet)?;
        let last = index + 1 == frames;
        let end = if last {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        let granule = if last { total } else { (index + 1) * FRAME } as u64;
        writer.write_packet(packet[..len].to_vec(), serial, end, granule)?;
    }
    Ok(writer.into_inner())
}
