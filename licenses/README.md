# SV Architecture Attribution

`src/models/sv/network.rs` implements the ERes2NetV2/AFF inference topology from
Alibaba 3D-Speaker, as adapted by GPT-SoVITS.

Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
Licensed under the Apache License, Version 2.0; see `3D-Speaker-Apache-2.0.txt`.

This version is implemented in Rust/Candle, evaluates only `forward3`, and omits
training and classification paths. Upstream source references are recorded in
`docs/SV.md`. No upstream model weights are distributed with this project.
