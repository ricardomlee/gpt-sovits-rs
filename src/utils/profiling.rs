use candle_core::Device;
use std::sync::OnceLock;

pub fn sync_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("GPT_SOVITS_SYNC_PROFILE")
            .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

pub fn sync_profile_stage(device: &Device) -> candle_core::Result<()> {
    if sync_profile_enabled() {
        device.synchronize()?;
    }
    Ok(())
}
