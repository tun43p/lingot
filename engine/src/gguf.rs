//! GGUF v3 reader: metadata, tensor descriptions and zero-copy tensor data over an mmap.

use std::fmt;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (only 3 is supported)")]
    Version(u32),
    #[error("truncated or corrupt GGUF file at offset {0}")]
    Truncated(usize),
    #[error("invalid GGUF file: {0}")]
    Invalid(String),
    #[error("unsupported tensor type id {0} (supported: F32, F16, BF16, Q8_0, Q4_0)")]
    TensorType(u32),
    #[error("missing or mistyped metadata key `{0}`")]
    Metadata(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    /// Any integer type, widened. Writers are free to pick the width.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) => v.try_into().ok(),
            Value::I16(v) => v.try_into().ok(),
            Value::I32(v) => v.try_into().ok(),
            Value::I64(v) => v.try_into().ok(),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Value::Bool(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::U8(v) => write!(f, "{v}"),
            Value::I8(v) => write!(f, "{v}"),
            Value::U16(v) => write!(f, "{v}"),
            Value::I16(v) => write!(f, "{v}"),
            Value::U32(v) => write!(f, "{v}"),
            Value::I32(v) => write!(f, "{v}"),
            Value::U64(v) => write!(f, "{v}"),
            Value::I64(v) => write!(f, "{v}"),
            Value::F32(v) => write!(f, "{v}"),
            Value::F64(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(s) => {
                // Long strings (chat template) are shortened for display.
                let short: String = s.chars().take(60).collect();
                let dots = if short.len() < s.len() { "..." } else { "" };
                write!(f, "{short:?}{dots}")
            }
            Value::Array(a) => {
                write!(f, "[{} items] [", a.len())?;
                for (i, v) in a.iter().take(6).enumerate() {
                    write!(f, "{}{v}", if i > 0 { ", " } else { "" })?;
                }
                write!(f, "{}]", if a.len() > 6 { ", ..." } else { "" })
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
}

impl TensorType {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => TensorType::F32,
            1 => TensorType::F16,
            2 => TensorType::Q4_0,
            8 => TensorType::Q8_0,
            30 => TensorType::BF16,
            other => return Err(Error::TensorType(other)),
        })
    }

    /// (elements per block, bytes per block)
    fn block(self) -> (u64, u64) {
        match self {
            TensorType::F32 => (1, 4),
            TensorType::F16 | TensorType::BF16 => (1, 2),
            TensorType::Q8_0 => (32, 34),
            TensorType::Q4_0 => (32, 18),
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGUF order: `dims[0]` is the innermost dimension (reversed from PyTorch).
    pub dims: Vec<u64>,
    pub ty: TensorType,
    /// Absolute byte range in the file.
    start: usize,
    len: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn file_offset(&self) -> usize {
        self.start
    }

    pub fn n_bytes(&self) -> usize {
        self.len
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    pub data_offset: usize,
    /// In file order.
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<TensorInfo>,
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        // SAFETY: the mapping is read-only; model files are not modified while loaded.
        let mmap = unsafe { Mmap::map(&file)? };
        let mut r = Reader { buf: &mmap, pos: 0 };

        if r.take(4)? != b"GGUF" {
            return Err(Error::BadMagic);
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(Error::Version(version));
        }
        let n_tensors = r.u64()?;
        let n_kv = r.u64()?;

        let mut metadata = Vec::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            metadata.push((key, r.value(ty)?));
        }

        let alignment = match metadata.iter().find(|(k, _)| k == "general.alignment") {
            Some((_, v)) => v
                .as_u64()
                .filter(|a| *a > 0)
                .ok_or_else(|| Error::Metadata("general.alignment".into()))?,
            None => 32,
        };

        // (name, dims, type, offset relative to the data section)
        let mut raw = Vec::new();
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()?;
            let dims = (0..n_dims).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
            let ty = TensorType::from_id(r.u32()?)?;
            raw.push((name, dims, ty, r.u64()?));
        }

        let data_offset = (r.pos as u64).next_multiple_of(alignment);
        let mut tensors = Vec::with_capacity(raw.len());
        for (name, dims, ty, rel) in raw {
            let invalid = || Error::Invalid(format!("tensor `{name}` has an invalid size or offset"));
            let (per_block, block_bytes) = ty.block();
            let n = dims
                .iter()
                .try_fold(1u64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(invalid)?;
            if n % per_block != 0 || rel % alignment != 0 {
                return Err(invalid());
            }
            let len = (n / per_block).checked_mul(block_bytes).ok_or_else(invalid)?;
            let start = data_offset.checked_add(rel).ok_or_else(invalid)?;
            let end = start.checked_add(len).ok_or_else(invalid)?;
            if end > mmap.len() as u64 {
                return Err(Error::Truncated(mmap.len()));
            }
            tensors.push(TensorInfo { name, dims, ty, start: start as usize, len: len as usize });
        }

        Ok(Gguf { mmap, version, alignment, data_offset: data_offset as usize, metadata, tensors })
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Like `get`, but a missing key is an error naming the key.
    pub fn require(&self, key: &str) -> Result<&Value> {
        self.get(key).ok_or_else(|| Error::Metadata(key.into()))
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Zero-copy view of a tensor's raw bytes.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.start..t.start + t.len]
    }
}

/// Hyperparameters, read only from metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub block_count: usize,
    pub context_length: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    pub head_count_kv: usize,
    pub key_length: usize,
    pub value_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f32,
}

impl Qwen3Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.require("general.architecture")?.as_str();
        if arch != Some("qwen3") {
            return Err(Error::Invalid(format!("architecture is {arch:?}, expected \"qwen3\"")));
        }
        let int = |key: &str| -> Result<usize> {
            let v = g.require(key)?.as_u64().ok_or_else(|| Error::Metadata(key.into()))?;
            Ok(v as usize)
        };
        let float = |key: &str| -> Result<f32> {
            g.require(key)?.as_f32().ok_or_else(|| Error::Metadata(key.into()))
        };
        let tokens = "tokenizer.ggml.tokens";
        Ok(Qwen3Config {
            vocab_size: g
                .require(tokens)?
                .as_array()
                .ok_or_else(|| Error::Metadata(tokens.into()))?
                .len(),
            block_count: int("qwen3.block_count")?,
            context_length: int("qwen3.context_length")?,
            embedding_length: int("qwen3.embedding_length")?,
            feed_forward_length: int("qwen3.feed_forward_length")?,
            head_count: int("qwen3.attention.head_count")?,
            head_count_kv: int("qwen3.attention.head_count_kv")?,
            key_length: int("qwen3.attention.key_length")?,
            value_length: int("qwen3.attention.value_length")?,
            rope_freq_base: float("qwen3.rope.freq_base")?,
            rms_norm_eps: float("qwen3.attention.layer_norm_rms_epsilon")?,
        })
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|e| *e <= self.buf.len());
        let end = end.ok_or(Error::Truncated(self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String> {
        let at = self.pos;
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| Error::Truncated(at))?;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::Invalid(format!("non UTF-8 string at offset {at}")))
    }

    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.array::<1>()?[0]),
            1 => Value::I8(self.array::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.array()?)),
            3 => Value::I16(i16::from_le_bytes(self.array()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.array()?)),
            6 => Value::F32(f32::from_le_bytes(self.array()?)),
            7 => Value::Bool(self.array::<1>()?[0] != 0),
            8 => Value::String(self.string()?),
            9 => {
                let item_ty = self.u32()?;
                if item_ty == 9 {
                    return Err(Error::Invalid("nested metadata arrays are not supported".into()));
                }
                let at = self.pos;
                let len = self.u64()?;
                // Every item takes at least one byte: reject absurd lengths before allocating.
                if len > (self.buf.len() - self.pos) as u64 {
                    return Err(Error::Truncated(at));
                }
                let items = (0..len).map(|_| self.value(item_ty)).collect::<Result<Vec<_>>>()?;
                Value::Array(items)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.array()?)),
            12 => Value::F64(f64::from_le_bytes(self.array()?)),
            other => return Err(Error::Invalid(format!("unknown metadata value type {other}"))),
        })
    }
}
