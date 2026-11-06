//! Model-free documentation checks. Bash is parsed, never executed; no network requests.
use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use std::{
    fs,
    path::{Path, PathBuf},
};

fn documents() -> Vec<PathBuf> {
    fn collect(directory: &Path, paths: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect(&path, paths);
            } else if path.extension().is_some_and(|extension| extension == "md") {
                paths.push(path);
            }
        }
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut paths = [
        "README.md",
        "IMPLEMENTATION.md",
        "site/README.md",
        "examples/samples/README.md",
        "experiments/cuda-oxide/README.md",
    ]
    .into_iter()
    .map(|path| root.join(path))
    .collect::<Vec<_>>();
    collect(&root.join("docs"), &mut paths);
    paths.sort();
    paths
}

fn blocks(markdown: &str) -> Vec<(String, String)> {
    let mut result = Vec::new();
    let mut current = None;
    for event in Parser::new_ext(markdown, Options::ENABLE_TABLES) {
        match event {
            Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(language))) => {
                current = Some((language.into_string(), String::new()));
            }
            Event::Text(text) => {
                if let Some((_, code)) = &mut current {
                    code.push_str(&text);
                }
            }
            Event::End(TagEnd::CodeBlock) => {
                if let Some(block) = current.take() {
                    result.push(block);
                }
            }
            _ => {}
        }
    }
    result
}

#[test]
fn local_markdown_link_targets_exist() {
    for path in documents() {
        let markdown = fs::read_to_string(&path).unwrap();
        for event in Parser::new_ext(&markdown, Options::ENABLE_TABLES) {
            let destination = match event {
                Event::Start(Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. }) => dest_url,
                _ => continue,
            };
            if destination.contains(':')
                || destination.starts_with('#')
                || destination.starts_with("//")
            {
                continue;
            }
            let target = destination.split(['#', '?']).next().unwrap();
            if target.is_empty() {
                continue;
            }
            assert!(
                path.parent().unwrap().join(target).exists(),
                "{}: broken local link {destination}",
                path.display()
            );
        }
    }
}

#[test]
fn json_examples_parse() {
    for path in documents() {
        for (language, code) in blocks(&fs::read_to_string(&path).unwrap()) {
            let values = match language.as_str() {
                "json" => vec![code.as_str()],
                "jsonl" => code
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .collect(),
                _ => continue,
            };
            for value in values {
                serde_json::from_str::<serde_json::Value>(value).unwrap_or_else(|error| {
                    panic!("{}: invalid {language}: {error}", path.display())
                });
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn bash_examples_are_syntax_checked_not_executed() {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    for path in documents() {
        for (language, code) in blocks(&fs::read_to_string(&path).unwrap()) {
            if !matches!(language.as_str(), "bash" | "sh") {
                continue;
            }
            let mut child = Command::new("bash")
                .arg("-n")
                .env_remove("BASH_ENV")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .expect("bash is required to check shell examples");
            child
                .stdin
                .take()
                .unwrap()
                .write_all(code.as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "{}: invalid shell syntax: {}",
                path.display(),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}

#[test]
fn documented_cli_options_exist_in_binary_help() {
    let help = std::process::Command::new(env!("CARGO_BIN_EXE_gpt-sovits"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    let markdown = include_str!("../docs/API.md");
    let section = markdown
        .split("## CLI\n")
        .nth(1)
        .unwrap()
        .split("## Rust API\n")
        .next()
        .unwrap();
    let mut checked = 0;
    for event in Parser::new_ext(section, Options::ENABLE_TABLES) {
        if let Event::Code(code) = event {
            if code.starts_with("--") {
                let flag = code.split_whitespace().next().unwrap();
                assert!(
                    help.split_whitespace()
                        .any(|word| word.trim_end_matches(',') == flag),
                    "API CLI table contains unsupported flag {flag}"
                );
                checked += 1;
            }
        }
    }
    assert!(checked >= 10, "CLI table was not parsed");
}

#[test]
fn fenced_block_parser_handles_json_lines_and_ignores_inline_examples() {
    let parsed = blocks("`inline`\n\n```jsonl\n{\"a\":1}\n{\"a\":2}\n```\n");
    assert_eq!(parsed, [("jsonl".into(), "{\"a\":1}\n{\"a\":2}\n".into())]);
}
