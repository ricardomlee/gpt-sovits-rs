use std::process::{Command, Output};

fn cli() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_gpt-sovits"));
    command.env_remove("GPT_SOVITS_HOST");
    command
}

fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn help_documents_loopback_default_and_environment_variable() {
    let output = cli().arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = diagnostics(&output);
    assert!(help.contains("--host <HOST>"), "{help}");
    assert!(help.contains("GPT_SOVITS_HOST"), "{help}");
    assert!(help.contains("default: 127.0.0.1"), "{help}");
}

#[test]
fn invalid_hosts_fail_during_argument_parsing() {
    for host in ["", " ", "127.0.0.1:9880", "not-an-ip", "[::1]", "999.1.1.1"] {
        let output = cli().args(["--http", "--host", host]).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{}", diagnostics(&output));
        assert!(diagnostics(&output).contains("--host"));
    }
}

#[test]
fn environment_host_is_validated_and_explicit_flag_takes_precedence() {
    let output = cli()
        .env("GPT_SOVITS_HOST", "invalid")
        .arg("--http")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2), "{}", diagnostics(&output));

    let dir = tempfile::tempdir().unwrap();
    for host in ["127.0.0.1", "0.0.0.0", "::1", "::"] {
        let output = cli()
            .env("GPT_SOVITS_HOST", "invalid")
            .args(["--host", host, "--list-voices", "--voices-dir"])
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", diagnostics(&output));
    }
}
