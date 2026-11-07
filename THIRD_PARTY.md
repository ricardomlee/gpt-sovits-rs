# Audio Codec Notices

The HTTP server uses LAME for MP3 encoding, libopus for Opus encoding, and
the Rust `ogg` crate for Ogg framing. No codec source is modified by this project.

- LAME: https://lame.sourceforge.io/ . The encoder library is distributed under
  LGPL-2.0-or-later, with LGPL-2.1-or-later components. We link the system shared
  library through `lame-sys` without its bundled/static feature. Binary archives
  carry the replaceable shared library; containers install the distribution's
  `libmp3lame0` package. Upstream source: https://sourceforge.net/projects/lame/files/lame/ .
  License texts and distribution copyright information are in `licenses/`.
- `lame-sys` 0.1.2: https://github.com/ebarnard/lame-sys . Its Rust bindings are
  offered under MIT/Apache-2.0; the LGPL bundled implementation is not enabled.
- libopus (`opusic-sys`): https://opus-codec.org/ . Built from the Cargo-locked
  bundled source under its BSD license, reproduced in `licenses/opus.txt`.
- Rust `opus`: https://github.com/SpaceManiac/opus-rs . MIT license in
  `licenses/opus-rs.txt`.
- Rust `ogg`: https://github.com/RustAudio/ogg . BSD license in `licenses/ogg.txt`.

Existing libsoxr remains dynamically linked. Its project and license information
are at https://sourceforge.net/projects/soxr/ ; the distribution copyright notice
is included in `licenses/soxr.txt`.

To replace a bundled LAME library, retain its ABI-compatible shared-library
filename beside the executable on macOS or in the archive's `lib/` directory on
Linux. Containers use the system library loader. Building this project from its
source and Cargo.lock also allows relinking against another compatible library.
