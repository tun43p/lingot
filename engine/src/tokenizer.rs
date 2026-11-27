//! Byte-level BPE tokenizer (GPT-2 style) with the Qwen2 pre-tokenizer, built from GGUF metadata.
//!
//! Encoding: special tokens are cut out whole, each remaining segment is NFC-normalized,
//! split by the Qwen2 regex (hand-written, see `match_at`), and each piece goes through BPE.

use std::borrow::Cow;
use std::collections::HashMap;

use crate::gguf::{Error, Gguf, Result};
use crate::unicode_data::{COMBINING_CLASS, COMPOSITION, DECOMPOSITION, LETTER};

// tokenizer.ggml.token_type values
const TYPE_NORMAL: u64 = 1;
const TYPE_CONTROL: u64 = 3;
const TYPE_USER_DEFINED: u64 = 4;

pub struct Tokenizer {
    /// id -> raw bytes of the token
    tokens: Vec<Vec<u8>>,
    /// id of the single-byte token for each byte
    byte_ids: [u32; 256],
    /// (left id, right id) -> (rank, merged id); lower rank merges first
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Control and user-defined tokens, longest first: matched whole, never through BPE.
    specials: Vec<(String, u32)>,
    nfc: bool,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let string = |key: &str| -> Result<&str> {
            g.require(key)?
                .as_str()
                .ok_or_else(|| Error::Metadata(key.into()))
        };
        let array = |key: &str| {
            g.require(key)?
                .as_array()
                .ok_or_else(|| Error::Metadata(key.into()))
        };
        let (model, pre) = (
            string("tokenizer.ggml.model")?,
            string("tokenizer.ggml.pre")?,
        );
        if (model, pre) != ("gpt2", "qwen2") {
            return Err(Error::Invalid(format!(
                "unsupported tokenizer {model}/{pre}, expected gpt2/qwen2"
            )));
        }
        // Qwen's tokenizer normalizes to NFC; files converted before the key existed lack it.
        let nfc = match g.get("tokenizer.ggml.normalizer.nfc") {
            Some(v) => v
                .as_bool()
                .ok_or_else(|| Error::Metadata("tokenizer.ggml.normalizer.nfc".into()))?,
            None => true,
        };

        let names = array("tokenizer.ggml.tokens")?;
        let types = array("tokenizer.ggml.token_type")?;
        if names.len() != types.len() {
            return Err(Error::Invalid("token and token_type counts differ".into()));
        }

        // GPT-2 byte <-> char mapping: printable bytes map to themselves, the rest to U+0100...
        let mut byte_char = ['\0'; 256];
        let mut next = 256u32;
        for b in 0..=255u8 {
            let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
            byte_char[b as usize] = if printable {
                b as char
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            };
        }
        let char_byte: HashMap<char, u8> =
            (0..=255u8).map(|b| (byte_char[b as usize], b)).collect();

        let mut tokens = Vec::with_capacity(names.len());
        let mut vocab: HashMap<&str, u32> = HashMap::new(); // normal tokens, in mapped form
        let mut specials = Vec::new();
        for (id, (name, ty)) in names.iter().zip(types).enumerate() {
            let bad = || Error::Invalid(format!("bad token {id} in tokenizer metadata"));
            let (name, ty) = (name.as_str().ok_or_else(bad)?, ty.as_u64().ok_or_else(bad)?);
            if ty == TYPE_NORMAL {
                let bytes: Option<Vec<u8>> =
                    name.chars().map(|c| char_byte.get(&c).copied()).collect();
                tokens.push(bytes.ok_or_else(bad)?);
                vocab.insert(name, id as u32);
            } else {
                tokens.push(name.as_bytes().to_vec());
                if (ty == TYPE_CONTROL || ty == TYPE_USER_DEFINED) && !name.is_empty() {
                    specials.push((name.to_string(), id as u32));
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let mut byte_ids = [0u32; 256];
        for b in 0..=255u8 {
            let mut buf = [0u8; 4];
            let key: &str = byte_char[b as usize].encode_utf8(&mut buf);
            byte_ids[b as usize] = *vocab
                .get(key)
                .ok_or_else(|| Error::Invalid(format!("vocabulary has no token for byte {b}")))?;
        }

        let mut merges = HashMap::new();
        for (rank, m) in array("tokenizer.ggml.merges")?.iter().enumerate() {
            let bad = || Error::Invalid(format!("bad merge {rank} in tokenizer metadata"));
            let (left, right) = m.as_str().and_then(|s| s.split_once(' ')).ok_or_else(bad)?;
            let id = |s: &str| vocab.get(s).copied().ok_or_else(bad);
            let merged = id(&format!("{left}{right}"))?;
            merges
                .entry((id(left)?, id(right)?))
                .or_insert((rank as u32, merged));
        }

        Ok(Tokenizer {
            tokens,
            byte_ids,
            merges,
            specials,
            nfc,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut plain_start = 0;
        let mut i = 0;
        while i < text.len() {
            // shortcut: linear scan over the specials at each position, use Aho-Corasick if
            // encoding ever shows up in a profile.
            let special = self
                .specials
                .iter()
                .find(|(s, _)| text[i..].starts_with(s.as_str()));
            if let Some((s, id)) = special {
                self.encode_plain(&text[plain_start..i], &mut ids);
                ids.push(*id);
                i += s.len();
                plain_start = i;
            } else {
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        self.encode_plain(&text[plain_start..], &mut ids);
        ids
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        let text = if self.nfc {
            nfc(text)
        } else {
            Cow::Borrowed(text)
        };
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let offset = |i: usize| chars.get(i).map_or(text.len(), |c| c.0);
        let mut i = 0;
        while i < chars.len() {
            let end = match_at(&chars, i);
            self.bpe(&text.as_bytes()[offset(i)..offset(end)], ids);
            i = end;
        }
    }

    /// Merge the lowest-rank adjacent pair, everywhere, until no pair can merge.
    fn bpe(&self, piece: &[u8], ids: &mut Vec<u32>) {
        let mut word: Vec<u32> = piece.iter().map(|b| self.byte_ids[*b as usize]).collect();
        loop {
            let best = word
                .windows(2)
                .filter_map(|w| self.merges.get(&(w[0], w[1])).map(|m| (*m, (w[0], w[1]))))
                .min();
            let Some(((_, merged), pair)) = best else {
                break;
            };
            let mut out = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && (word[i], word[i + 1]) == pair {
                    out.push(merged);
                    i += 2;
                } else {
                    out.push(word[i]);
                    i += 1;
                }
            }
            word = out;
        }
        ids.extend(word);
    }

    /// Raw bytes of a token. Panics if `id` is not in the vocabulary.
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        &self.tokens[id as usize]
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut stream = self.stream();
        let mut out: String = ids.iter().map(|id| stream.push(*id)).collect();
        out.push_str(&stream.finish());
        out
    }

    pub fn stream(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tokenizer: self,
            pending: Vec::new(),
        }
    }
}

/// Decodes token by token without ever splitting a UTF-8 character: bytes of an incomplete
/// character are held back until the next token completes it.
pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    pending: Vec<u8>,
}

impl StreamDecoder<'_> {
    pub fn push(&mut self, id: u32) -> String {
        self.pending
            .extend_from_slice(self.tokenizer.token_bytes(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        // Invalid bytes: one replacement character, like String::from_utf8_lossy.
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + n);
                        }
                        // Incomplete character at the end: wait for the next token.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// Flush what is left at the end of the stream (an unfinished character becomes U+FFFD).
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

fn is_letter(c: char) -> bool {
    let c = c as u32;
    LETTER
        .binary_search_by(|&(lo, hi)| {
            if hi < c {
                std::cmp::Ordering::Less
            } else if lo > c {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

/// End (exclusive, in chars) of the Qwen2 pre-tokenizer match starting at char `i`:
///
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
///
/// Alternatives are tried in order, as a backtracking regex engine does. Every char starts a match.
fn match_at(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    let at = |j: usize| chars.get(j).map(|c| c.1);
    let is = |j: usize, f: fn(char) -> bool| at(j).is_some_and(f);
    let newline = |c: char| c == '\r' || c == '\n';
    let other = |c: char| !c.is_whitespace() && !is_letter(c) && !c.is_numeric();
    let c = chars[i].1;

    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    // shortcut: ASCII case folding only (the regex also folds U+017F to s), revisit if a
    // corpus mismatch ever points here.
    if c == '\'' {
        let a = at(i + 1).map(|c| c.to_ascii_lowercase());
        let b = at(i + 2).map(|c| c.to_ascii_lowercase());
        match (a, b) {
            (Some('s' | 't' | 'm' | 'd'), _) => return i + 2,
            (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l')) => return i + 3,
            _ => {}
        }
    }

    // [^\r\n\p{L}\p{N}]?\p{L}+
    let letters = if is_letter(c) {
        Some(i)
    } else if !newline(c) && !c.is_numeric() && is(i + 1, is_letter) {
        Some(i + 1)
    } else {
        None
    };
    if let Some(mut end) = letters {
        while is(end, is_letter) {
            end += 1;
        }
        return end;
    }

    // \p{N}
    if c.is_numeric() {
        return i + 1;
    }

    //  ?[^\s\p{L}\p{N}]+[\r\n]*
    let start = if c == ' ' { i + 1 } else { i };
    let mut end = start;
    while is(end, other) {
        end += 1;
    }
    if end > start {
        while is(end, newline) {
            end += 1;
        }
        return end;
    }

    // Only whitespace can reach this point. `end` = end of the whitespace run.
    let mut end = i;
    while is(end, char::is_whitespace) {
        end += 1;
    }
    // \s*[\r\n]+ : up to the last newline of the run
    if let Some(last) = (i..end).rev().find(|j| is(*j, newline)) {
        return last + 1;
    }
    // \s+(?!\S) : the whole run at end of text, else all but the last whitespace
    if end == n {
        return end;
    }
    if end - i >= 2 {
        return end - 1;
    }
    // \s+
    end
}

// Hangul syllables are composed and decomposed arithmetically (Unicode chapter 3.12).
const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const S_COUNT: u32 = L_COUNT * V_COUNT * T_COUNT;

fn combining_class(c: char) -> u8 {
    let c = c as u32;
    let i = COMBINING_CLASS.partition_point(|&(_, hi, _)| hi < c);
    match COMBINING_CLASS.get(i) {
        Some(&(lo, _, class)) if lo <= c => class,
        _ => 0,
    }
}

fn decompose(c: char, out: &mut Vec<char>) {
    let cp = c as u32;
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let s = cp - S_BASE;
        out.push(char::from_u32(L_BASE + s / (V_COUNT * T_COUNT)).unwrap());
        out.push(char::from_u32(V_BASE + s % (V_COUNT * T_COUNT) / T_COUNT).unwrap());
        if !s.is_multiple_of(T_COUNT) {
            out.push(char::from_u32(T_BASE + s % T_COUNT).unwrap());
        }
    } else if let Ok(i) = DECOMPOSITION.binary_search_by_key(&cp, |d| d.0) {
        let (_, first, second) = DECOMPOSITION[i];
        decompose(char::from_u32(first).unwrap(), out);
        if second != 0 {
            decompose(char::from_u32(second).unwrap(), out);
        }
    } else {
        out.push(c);
    }
}

fn compose(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return char::from_u32(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE).is_multiple_of(T_COUNT)
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return char::from_u32(a + b - T_BASE);
    }
    let i = COMPOSITION
        .binary_search_by_key(&(a, b), |c| (c.0, c.1))
        .ok()?;
    char::from_u32(COMPOSITION[i].2)
}

/// Unicode Normalization Form C.
fn nfc(text: &str) -> Cow<'_, str> {
    // Everything below U+0300 is a starter with no decomposition into combining marks.
    if text.chars().all(|c| c < '\u{300}') {
        return Cow::Borrowed(text);
    }

    let mut chars = Vec::new();
    for c in text.chars() {
        decompose(c, &mut chars);
    }

    // Canonical ordering: stable sort of each run of non-starters by combining class.
    let mut i = 0;
    while i < chars.len() {
        if combining_class(chars[i]) == 0 {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < chars.len() && combining_class(chars[end]) != 0 {
            end += 1;
        }
        chars[i..end].sort_by_key(|c| combining_class(*c));
        i = end;
    }

    // Canonical composition.
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut starter: Option<usize> = None;
    let mut last_class = 0;
    for c in chars {
        let class = combining_class(c);
        if let Some(s) = starter {
            // `c` is blocked from the starter by a previous mark of the same or higher class.
            let adjacent = s == out.len() - 1;
            if (adjacent || (last_class != 0 && last_class < class))
                && let Some(composed) = compose(out[s], c)
            {
                out[s] = composed;
                continue;
            }
        }
        if class == 0 {
            starter = Some(out.len());
        }
        last_class = class;
        out.push(c);
    }
    Cow::Owned(out.into_iter().collect())
}
