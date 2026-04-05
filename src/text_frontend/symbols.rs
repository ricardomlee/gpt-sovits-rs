//! Phoneme Symbol Table

use crate::Result;
use std::collections::HashMap;

/// Symbol table for phoneme encoding/decoding
#[derive(Debug)]
pub struct SymbolTable {
    symbol_to_id: HashMap<String, usize>,
    id_to_symbol: Vec<String>,
    pad_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl SymbolTable {
    /// Create a new symbol table with default GPT-SoVITS symbols
    pub fn new() -> Self {
        // Default phoneme symbols for GPT-SoVITS
        // This is a simplified version - the full table has ~1000 symbols
        let symbols: Vec<&str> = vec![
            "_",           // padding
            "^",           // beginning of sentence
            "$",           // end of sentence
            " ",           // word separator
            // Chinese initials
            "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
            "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s",
            // Chinese finals
            "a", "o", "e", "i", "u", "v",
            "ai", "ei", "ui", "ao", "ou", "iu", "ie", "ue", "er",
            "an", "en", "in", "un", "vn",
            "ang", "eng", "ing", "ong", "iong",
            // English phonemes (simplified)
            "æ", "ʌ", "ɑ", "ɔ", "ʊ", "u", "ɛ", "ɪ", "i", "ə",
            "θ", "ð", "ʃ", "ʒ", "tʃ", "dʒ", "tr", "dr", "ts", "dz",
            // Japanese
            "a", "i", "u", "e", "o", "k", "s", "t", "n", "h",
            "m", "y", "r", "w", "g", "z", "d", "b", "p", "N", "Q", "V",
        ];

        let symbol_to_id: HashMap<String, usize> = symbols
            .iter()
            .enumerate()
            .map(|(i, &s)| (s.to_string(), i))
            .collect();

        let id_to_symbol: Vec<String> = symbols.iter().map(|&s| s.to_string()).collect();

        Self {
            pad_id: 0,
            bos_id: 1,
            eos_id: 2,
            symbol_to_id,
            id_to_symbol,
        }
    }

    /// Encode phoneme string to IDs
    pub fn encode(&self, phonemes: &str) -> Result<Vec<usize>> {
        // Add BOS token
        let mut ids = vec![self.bos_id];

        // Simple tokenization - in production would need proper segmentation
        for ch in phonemes.chars() {
            let symbol = ch.to_string();
            if let Some(&id) = self.symbol_to_id.get(&symbol) {
                ids.push(id);
            } else {
                // Unknown symbol - use pad
                ids.push(self.pad_id);
            }
        }

        // Add EOS token
        ids.push(self.eos_id);

        Ok(ids)
    }

    /// Decode IDs to phoneme string
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let phonemes: String = ids
            .iter()
            .filter_map(|&id| {
                if id == self.pad_id || id == self.bos_id || id == self.eos_id {
                    None
                } else {
                    self.id_to_symbol.get(id).cloned()
                }
            })
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

    /// Get padding token ID
    pub fn pad_id(&self) -> usize {
        self.pad_id
    }

    /// Get BOS token ID
    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    /// Get EOS token ID
    pub fn eos_id(&self) -> usize {
        self.eos_id
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
    fn test_encode_decode() {
        let table = SymbolTable::new();
        let ids = table.encode("a o e").unwrap();
        let decoded = table.decode(&ids).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_symbol_table_len() {
        let table = SymbolTable::new();
        assert!(table.len() > 0);
    }
}
