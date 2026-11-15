//! Phoneme Symbol Table
//!
//! Matches GPT-SoVITS v2 symbol table (732 symbols).
//! Loaded from embedded JSON at compile time to guarantee exact match with Python.

use crate::Result;
use std::collections::HashMap;

/// All v2 symbols as a JSON array, embedded at compile time.
/// Generated from Python's `text.symbols_v2.symbols`.
const SYMBOLS_JSON: &str = include_str!("symbols_v2.json");

/// Symbol table for phoneme encoding/decoding
#[derive(Debug)]
pub struct SymbolTable {
    symbol_to_id: HashMap<String, usize>,
    id_to_symbol: Vec<String>,
    max_symbol_bytes: usize,
}

impl SymbolTable {
    /// Create a new symbol table matching GPT-SoVITS v2 symbols
    pub fn new() -> Self {
        let symbols: Vec<String> =
            serde_json::from_str(SYMBOLS_JSON).expect("Failed to parse embedded symbols JSON");

        let symbol_to_id: HashMap<String, usize> = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Self {
            max_symbol_bytes: symbols.iter().map(String::len).max().unwrap_or(0),
            symbol_to_id,
            id_to_symbol: symbols,
        }
    }

    /// Encode phoneme string to IDs
    /// Input format: space-separated phonemes like "n i3 h ao3 sh ir4"
    /// Spaces are skipped. Uses longest-match-first for multi-char symbols.
    pub fn encode(&self, phonemes: &str) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        let mut remaining = phonemes;
        while !remaining.is_empty() {
            if let Some(rest) = remaining.strip_prefix(' ') {
                remaining = rest;
                continue;
            }
            let best_match = (1..=remaining.len().min(self.max_symbol_bytes))
                .rev()
                .find_map(|len| {
                    let symbol = remaining.get(..len)?;
                    self.symbol_to_id.get(symbol).map(|&id| (len, id))
                });
            let consumed = if let Some((len, id)) = best_match {
                ids.push(id);
                len
            } else {
                remaining.chars().next().unwrap().len_utf8()
            };
            remaining = &remaining[consumed..];
        }

        Ok(ids)
    }

    /// Decode IDs to phoneme string
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let phonemes: String = ids
            .iter()
            .filter_map(|&id| self.id_to_symbol.get(id))
            .cloned()
            .collect();

        Ok(phonemes)
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Check if symbol table is empty
    pub fn is_empty(&self) -> bool {
        self.id_to_symbol.is_empty()
    }

    /// Get symbol by ID
    pub fn get_symbol(&self, id: usize) -> Option<&str> {
        self.id_to_symbol.get(id).map(|s| s.as_str())
    }

    /// Get ID by symbol
    pub fn get_id(&self, symbol: &str) -> Option<usize> {
        self.symbol_to_id.get(symbol).copied()
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_symbol_encodes_to_its_trained_id() {
        let table = SymbolTable::new();
        for (id, symbol) in table.id_to_symbol.iter().enumerate() {
            assert_eq!(table.encode(symbol).unwrap(), vec![id], "{symbol}");
        }
    }

    #[test]
    fn greedy_encoding_matches_full_vocabulary_scan() {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let table = SymbolTable::new();
        let full_scan = |input: &str| {
            let chars: Vec<_> = input.chars().collect();
            let mut pos = 0;
            let mut ids = Vec::new();
            while pos < chars.len() {
                if chars[pos] == ' ' {
                    pos += 1;
                    continue;
                }
                let remaining: String = chars[pos..].iter().collect();
                let best = table
                    .symbol_to_id
                    .iter()
                    .filter(|(symbol, _)| remaining.starts_with(symbol.as_str()))
                    .max_by_key(|(symbol, _)| symbol.len());
                if let Some((symbol, &id)) = best {
                    ids.push(id);
                    pos += symbol.chars().count();
                } else {
                    pos += 1;
                }
            }
            ids
        };
        for input in [
            String::new(),
            table.id_to_symbol.join(" "),
            table.id_to_symbol.concat(),
            "sh ao3\t🙂你好\n\0a\u{301}".into(),
        ] {
            assert_eq!(table.encode(&input).unwrap(), full_scan(&input), "{input}");
        }
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..256 {
            let mut input = String::new();
            for _ in 0..16 {
                input.push_str(&table.id_to_symbol[rng.random_range(0..table.len())]);
                input.push_str(
                    ["", " ", "\t", "\n", "🙂", "你", "\0", "\u{301}"][rng.random_range(0..8)],
                );
            }
            assert_eq!(table.encode(&input).unwrap(), full_scan(&input), "{input}");
        }
    }

    #[test]
    fn test_encode_chinese() {
        let table = SymbolTable::new();
        // Test encoding space-separated initials+finals
        let ids = table.encode("n i3 h ao3 sh ir4 j ie4").unwrap();
        assert_eq!(ids.len(), 8);
        // Verify specific IDs match Python v2
        assert_eq!(table.get_id("n"), Some(227));
        assert_eq!(table.get_id("i3"), Some(168));
        assert_eq!(table.get_id("ao3"), Some(119));
        assert_eq!(table.get_id("sh"), Some(251));
        assert_eq!(table.get_id("ir4"), Some(214));
        assert_eq!(table.get_id("j"), Some(221));
        assert_eq!(table.get_id("ie4"), Some(194));
    }

    #[test]
    fn test_symbol_table_len() {
        let table = SymbolTable::new();
        assert_eq!(table.len(), 732);
    }
}
