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
}

impl SymbolTable {
    /// Create a new symbol table matching GPT-SoVITS v2 symbols
    pub fn new() -> Self {
        let symbols: Vec<String> = serde_json::from_str(SYMBOLS_JSON)
            .expect("Failed to parse embedded symbols JSON");

        let symbol_to_id: HashMap<String, usize> = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i))
            .collect();

        Self {
            symbol_to_id,
            id_to_symbol: symbols,
        }
    }

    /// Encode phoneme string to IDs
    /// Input format: space-separated phonemes like "n i3 h ao3 sh ir4"
    /// Spaces are skipped. Uses longest-match-first for multi-char symbols.
    pub fn encode(&self, phonemes: &str) -> Result<Vec<usize>> {
        let mut ids = Vec::new();
        let chars: Vec<char> = phonemes.chars().collect();
        let mut pos = 0;

        while pos < chars.len() {
            // Skip spaces
            if chars[pos] == ' ' {
                pos += 1;
                continue;
            }

            let mut matched = false;

            // Try longest symbols first (sorted by length descending)
            // We iterate through all symbols and pick the longest match
            let remaining: String = chars[pos..].iter().collect();
            let mut best_match: Option<(&str, usize)> = None;

            for (symbol, &id) in &self.symbol_to_id {
                if remaining.starts_with(symbol.as_str()) {
                    if let Some((prev_sym, _)) = best_match {
                        if symbol.len() > prev_sym.len() {
                            best_match = Some((symbol, id));
                        }
                    } else {
                        best_match = Some((symbol, id));
                    }
                }
            }

            if let Some((symbol, id)) = best_match {
                ids.push(id);
                pos += symbol.chars().count();
                matched = true;
            }

            if !matched {
                // Unknown symbol - skip
                pos += 1;
            }
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
