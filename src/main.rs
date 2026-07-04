//! GPT-SoVITS CLI binary entry point.

mod cli;
mod doctor;

#[cfg(feature = "http-api")]
mod server;

fn main() {
    cli::run();
}
