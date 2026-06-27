//! Conventional model layout discovery for user-facing tools.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct ModelPathOverrides {
    pub gpt: Option<PathBuf>,
    pub sovits: Option<PathBuf>,
    pub bert: Option<PathBuf>,
    pub hubert: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPaths {
    pub gpt: PathBuf,
    pub sovits: PathBuf,
    pub bert: Option<PathBuf>,
    pub hubert: Option<PathBuf>,
}

impl ModelPaths {
    pub fn discover(models_dir: &Path, overrides: ModelPathOverrides) -> Result<Self, String> {
        Ok(Self {
            gpt: resolve_required(
                "GPT",
                models_dir,
                overrides.gpt,
                &["gpt-model.safetensors"],
                "--gpt-model",
            )?,
            sovits: resolve_required(
                "SoVITS",
                models_dir,
                overrides.sovits,
                &["sovits-model.safetensors"],
                "--sovits-model",
            )?,
            bert: resolve_optional(
                "BERT",
                models_dir,
                overrides.bert,
                &["bert/bert.safetensors", "bert.safetensors"],
            )?,
            hubert: resolve_optional(
                "HuBERT",
                models_dir,
                overrides.hubert,
                &["hubert/hubert.safetensors", "hubert.safetensors"],
            )?,
        })
    }
}

fn resolve_required(
    name: &str,
    models_dir: &Path,
    explicit: Option<PathBuf>,
    candidates: &[&str],
    flag: &str,
) -> Result<PathBuf, String> {
    if let Some(path) = resolve_optional(name, models_dir, explicit, candidates)? {
        return Ok(path);
    }

    let expected = models_dir.join(candidates[0]);
    Err(format!(
        "{name} model not found. Put it at {} or pass {flag} <PATH>",
        expected.display()
    ))
}

fn resolve_optional(
    name: &str,
    models_dir: &Path,
    explicit: Option<PathBuf>,
    candidates: &[&str],
) -> Result<Option<PathBuf>, String> {
    if let Some(path) = explicit {
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(format!(
            "{name} model file does not exist: {}",
            path.display()
        ));
    }

    Ok(candidates
        .iter()
        .map(|candidate| models_dir.join(candidate))
        .find(|path| path.is_file()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, b"test").unwrap();
    }

    #[test]
    fn discovers_standard_nested_layout() {
        let temp = tempfile::tempdir().unwrap();
        touch(&temp.path().join("gpt-model.safetensors"));
        touch(&temp.path().join("sovits-model.safetensors"));
        touch(&temp.path().join("bert/bert.safetensors"));
        touch(&temp.path().join("hubert/hubert.safetensors"));

        let paths = ModelPaths::discover(temp.path(), ModelPathOverrides::default()).unwrap();

        assert_eq!(paths.gpt, temp.path().join("gpt-model.safetensors"));
        assert_eq!(paths.sovits, temp.path().join("sovits-model.safetensors"));
        assert_eq!(paths.bert, Some(temp.path().join("bert/bert.safetensors")));
        assert_eq!(
            paths.hubert,
            Some(temp.path().join("hubert/hubert.safetensors"))
        );
    }

    #[test]
    fn accepts_flat_optional_model_layout() {
        let temp = tempfile::tempdir().unwrap();
        touch(&temp.path().join("gpt-model.safetensors"));
        touch(&temp.path().join("sovits-model.safetensors"));
        touch(&temp.path().join("bert.safetensors"));
        touch(&temp.path().join("hubert.safetensors"));

        let paths = ModelPaths::discover(temp.path(), ModelPathOverrides::default()).unwrap();

        assert_eq!(paths.bert, Some(temp.path().join("bert.safetensors")));
        assert_eq!(paths.hubert, Some(temp.path().join("hubert.safetensors")));
    }

    #[test]
    fn explicit_paths_take_precedence() {
        let temp = tempfile::tempdir().unwrap();
        let custom_gpt = temp.path().join("custom/gpt.safetensors");
        let custom_sovits = temp.path().join("custom/sovits.safetensors");
        touch(&custom_gpt);
        touch(&custom_sovits);

        let paths = ModelPaths::discover(
            temp.path(),
            ModelPathOverrides {
                gpt: Some(custom_gpt.clone()),
                sovits: Some(custom_sovits.clone()),
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(paths.gpt, custom_gpt);
        assert_eq!(paths.sovits, custom_sovits);
    }

    #[test]
    fn missing_required_model_has_actionable_error() {
        let temp = tempfile::tempdir().unwrap();
        let error = ModelPaths::discover(temp.path(), ModelPathOverrides::default()).unwrap_err();

        assert!(error.contains("gpt-model.safetensors"));
        assert!(error.contains("--gpt-model"));
    }

    #[test]
    fn rejects_missing_explicit_optional_model() {
        let temp = tempfile::tempdir().unwrap();
        touch(&temp.path().join("gpt-model.safetensors"));
        touch(&temp.path().join("sovits-model.safetensors"));

        let error = ModelPaths::discover(
            temp.path(),
            ModelPathOverrides {
                bert: Some(temp.path().join("missing.safetensors")),
                ..Default::default()
            },
        )
        .unwrap_err();

        assert!(error.contains("BERT model file does not exist"));
    }
}
