//! Metal backend. Every operation is encoded into the current command buffer; nothing runs
//! until `read`, which commits it and waits. A decode step is therefore one command buffer
//! with no CPU round trip, and only the logits come back.
//!
//! Buffers use shared storage (unified memory): weights are the mmapped file itself, wrapped
//! without a copy, and activations are read back by plain pointer access.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use super::{AttentionShape, Backend, Weight};
use crate::gguf::TensorType;

// MTLCreateSystemDefaultDevice returns nil in a command-line tool unless CoreGraphics is linked.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

unsafe extern "C" {
    fn getpagesize() -> i32;
}

const SHADERS: &str = include_str!("../../shaders/kernels.metal");
const KERNELS: &[&str] = &[
    "embed_f32",
    "embed_bf16",
    "embed_q8_0",
    "embed_q4_0",
    "matmul_f32",
    "matmul_bf16",
    "matmul_q8_0",
    "matmul_q4_0",
    "matmul_tiled_f32",
    "matmul_tiled_bf16",
    "matmul_tiled_q8_0",
    "matmul_tiled_q4_0",
    "rms_norm",
    "rms_norm_any",
    "rope",
    "copy_at",
    "attention_scores",
    "attention_softmax",
    "attention_mix",
    "silu_mul",
    "add",
];

type MtlBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// Threads per simdgroup the matmul kernels are written for.
const SIMD_WIDTH: usize = 32;
/// Batch size from which the tiled matmul kernel is used.
const TILED_MIN_TOKENS: usize = 8;

/// Rows computed by one simdgroup in the matmul kernel of each weight type (the argument of
/// `MATMUL_BEGIN` in kernels.metal).
fn matmul_rows(ty: TensorType) -> usize {
    match ty {
        TensorType::Q8_0 | TensorType::Q4_0 => 8,
        _ => 4,
    }
}

/// What a point of a dispatch grid is.
#[derive(Clone, Copy)]
enum Grid {
    /// One thread.
    Threads,
    /// One threadgroup of this many threads.
    Groups(usize),
}

fn size(grid: [usize; 3]) -> MTLSize {
    MTLSize {
        width: grid[0],
        height: grid[1],
        depth: grid[2],
    }
}

/// Must match `Params` in kernels.metal.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Params {
    rows: u32,
    cols: u32,
    n: u32,
    head_dim: u32,
    n_head: u32,
    n_head_kv: u32,
    pos: u32,
    len: u32,
    eps: f32,
    scale: f32,
}

/// `len` f32 values starting `offset` bytes into a Metal buffer.
pub struct Buf {
    buffer: MtlBuffer,
    offset: usize,
    len: usize,
}

/// A weight matrix inside a no-copy buffer over the mmapped file.
pub struct Weights<'a> {
    buffer: MtlBuffer,
    offset: usize,
    ty: TensorType,
    rows: usize,
    cols: usize,
    _file: PhantomData<&'a [u8]>,
}

struct Pending {
    commands: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

pub struct Metal {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    /// Command buffer being encoded, if any.
    pending: RefCell<Option<Pending>>,
    /// With LINGOT_PROFILE set: (dispatches, GPU seconds) per kernel, printed on drop. Each
    /// dispatch then runs in its own command buffer, so only the GPU times are meaningful.
    profile: Option<RefCell<HashMap<String, (usize, f64)>>>,
    /// RoPE frequencies, cached per (head_dim, freq_base bits).
    inv_freq: RefCell<Option<(usize, u32, MtlBuffer)>>,
}

impl Metal {
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device")?;
        let queue = device
            .newCommandQueue()
            .ok_or("cannot create a Metal command queue")?;
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(SHADERS), None)
            .map_err(|e| format!("shader compilation failed: {e}"))?;
        let mut pipelines = HashMap::new();
        for name in KERNELS {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| format!("kernel `{name}` not found in the shader library"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("pipeline for `{name}` failed: {e}"))?;
            if pipeline.threadExecutionWidth() != SIMD_WIDTH
                || pipeline.maxTotalThreadsPerThreadgroup() < 4 * SIMD_WIDTH
            {
                return Err(format!(
                    "kernel `{name}`: this GPU runs {} threads per simdgroup and {} per \
                     threadgroup, the kernels expect {SIMD_WIDTH} and at least {}",
                    pipeline.threadExecutionWidth(),
                    pipeline.maxTotalThreadsPerThreadgroup(),
                    4 * SIMD_WIDTH
                ));
            }
            pipelines.insert(*name, pipeline);
        }
        Ok(Metal {
            device,
            queue,
            pipelines,
            pending: RefCell::new(None),
            profile: std::env::var_os("LINGOT_PROFILE").map(|_| RefCell::default()),
            inv_freq: RefCell::new(None),
        })
    }

    /// Encode one kernel: `params` at buffer index 0, `buffers` (with byte offsets) from 1,
    /// one thread per point of `grid`.
    fn dispatch(
        &self,
        kernel: &str,
        params: Params,
        buffers: &[(&MtlBuffer, usize)],
        groups: Grid,
        grid: [usize; 3],
    ) {
        assert!(grid.iter().all(|g| *g > 0), "{kernel}: empty dispatch");
        let mut slot = self.pending.borrow_mut();
        let pending = slot.get_or_insert_with(|| {
            let commands = self.queue.commandBuffer().expect("Metal command buffer");
            let encoder = commands.computeCommandEncoder().expect("Metal encoder");
            Pending { commands, encoder }
        });
        let pipeline = &self.pipelines[kernel];
        let encoder = &pending.encoder;
        encoder.setComputePipelineState(pipeline);
        // SAFETY: `params` outlives the call (the bytes are copied), buffers are alive and the
        // offsets are within them.
        unsafe {
            encoder.setBytes_length_atIndex(
                NonNull::from(&params).cast::<c_void>(),
                size_of::<Params>(),
                0,
            );
            for (i, (buffer, offset)) in buffers.iter().enumerate() {
                encoder.setBuffer_offset_atIndex(Some(buffer), *offset, i + 1);
            }
        }
        match groups {
            // One thread per grid point.
            Grid::Threads => {
                let width = pipeline.threadExecutionWidth().min(grid[0]);
                encoder.dispatchThreads_threadsPerThreadgroup(size(grid), size([width, 1, 1]));
            }
            Grid::Groups(threads) => {
                encoder
                    .dispatchThreadgroups_threadsPerThreadgroup(size(grid), size([threads, 1, 1]));
            }
        }
        drop(slot);
        if let Some(profile) = &self.profile {
            let seconds = self.flush();
            let mut profile = profile.borrow_mut();
            let entry = profile.entry(kernel.to_string()).or_default();
            *entry = (entry.0 + 1, entry.1 + seconds);
        }
    }

    /// Run everything encoded so far and wait for it. Returns the time the GPU spent on it.
    fn flush(&self) -> f64 {
        let Some(pending) = self.pending.borrow_mut().take() else {
            return 0.0;
        };
        pending.encoder.endEncoding();
        pending.commands.commit();
        pending.commands.waitUntilCompleted();
        if let Some(error) = pending.commands.error() {
            panic!("Metal command buffer failed: {error}");
        }
        pending.commands.GPUEndTime() - pending.commands.GPUStartTime()
    }

    fn buffer_with_bytes(&self, bytes: &[u8]) -> MtlBuffer {
        // SAFETY: Metal copies `bytes.len()` bytes from a valid pointer.
        unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new(bytes.as_ptr().cast_mut().cast()).unwrap(),
                bytes.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Metal buffer allocation failed")
    }

    fn rope_frequencies(&self, head_dim: usize, freq_base: f32) -> MtlBuffer {
        let mut cache = self.inv_freq.borrow_mut();
        match &*cache {
            Some((d, base, buffer)) if (*d, *base) == (head_dim, freq_base.to_bits()) => {
                buffer.clone()
            }
            _ => {
                // Same f32 arithmetic as the CPU backend, so both rotate by the same angles.
                let freqs: Vec<u8> = (0..head_dim / 2)
                    .map(|i| 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32))
                    .flat_map(f32::to_le_bytes)
                    .collect();
                let buffer = self.buffer_with_bytes(&freqs);
                *cache = Some((head_dim, freq_base.to_bits(), buffer.clone()));
                buffer
            }
        }
    }
}

impl Drop for Metal {
    fn drop(&mut self) {
        let Some(profile) = &self.profile else { return };
        let mut rows: Vec<_> = profile.borrow().clone().into_iter().collect();
        rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let total: f64 = rows.iter().map(|r| r.1.1).sum();
        eprintln!(
            "{:<20} {:>9} {:>10} {:>7} {:>10}",
            "kernel", "calls", "GPU s", "%", "µs/call"
        );
        for (name, (calls, seconds)) in rows {
            eprintln!(
                "{name:<20} {calls:>9} {seconds:>10.3} {:>6.1}% {:>10.1}",
                100.0 * seconds / total,
                1e6 * seconds / calls as f64
            );
        }
    }
}

fn kernel_suffix(ty: TensorType) -> &'static str {
    match ty {
        TensorType::F32 => "f32",
        TensorType::BF16 => "bf16",
        TensorType::Q8_0 => "q8_0",
        TensorType::Q4_0 => "q4_0",
        TensorType::F16 => panic!("Metal backend does not support F16 weights"),
    }
}

impl Backend for Metal {
    type Buf = Buf;
    type Weights<'a> = Weights<'a>;

    fn weights<'a>(&self, w: Weight<'a>) -> Weights<'a> {
        // No-copy buffers must start on a page and cover whole pages: wrap the pages that
        // contain the tensor and remember where it starts inside them. The file is mmapped
        // from a page boundary, so those pages are all mapped.
        let page = unsafe { getpagesize() } as usize;
        let address = w.data.as_ptr() as usize;
        let start = address / page * page;
        let offset = address - start;
        let length = (offset + w.data.len()).next_multiple_of(page);
        // SAFETY: the pages are mapped for at least 'a (the lifetime of the returned value),
        // kernels only read them, and no deallocator is installed.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    NonNull::new(start as *mut c_void).unwrap(),
                    length,
                    MTLResourceOptions::StorageModeShared,
                    None,
                )
        }
        .expect("Metal refused to wrap the model file without a copy");
        Weights {
            buffer,
            offset,
            ty: w.ty,
            rows: w.rows,
            cols: w.cols,
            _file: PhantomData,
        }
    }

    fn alloc(&self, len: usize) -> Buf {
        // New Metal buffers are zero-filled.
        let buffer = self
            .device
            .newBufferWithLength_options(len.max(1) * 4, MTLResourceOptions::StorageModeShared)
            .expect("Metal buffer allocation failed");
        Buf {
            buffer,
            offset: 0,
            len,
        }
    }

    fn upload(&self, data: &[f32]) -> Buf {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Buf {
            buffer: self.buffer_with_bytes(&bytes),
            offset: 0,
            len: data.len(),
        }
    }

    fn read(&self, buf: &Buf) -> Vec<f32> {
        self.flush();
        // SAFETY: shared-storage buffer of at least offset + len * 4 bytes, f32-aligned, and
        // no GPU work is pending after the flush.
        unsafe {
            let start = buf
                .buffer
                .contents()
                .cast::<u8>()
                .add(buf.offset)
                .cast::<f32>();
            std::slice::from_raw_parts(start.as_ptr(), buf.len).to_vec()
        }
    }

    fn slice(&self, buf: &Buf, start: usize, len: usize) -> Buf {
        assert!(start + len <= buf.len);
        Buf {
            buffer: buf.buffer.clone(),
            offset: buf.offset + start * 4,
            len,
        }
    }

    fn embed(&self, w: &Weights, tokens: &[u32], out: &mut Buf) {
        assert!(out.len == tokens.len() * w.cols && tokens.iter().all(|t| (*t as usize) < w.rows));
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let tokens_buffer = self.buffer_with_bytes(&bytes);
        let params = Params {
            cols: w.cols as u32,
            ..Default::default()
        };
        self.dispatch(
            &format!("embed_{}", kernel_suffix(w.ty)),
            params,
            &[
                (&w.buffer, w.offset),
                (&tokens_buffer, 0),
                (&out.buffer, out.offset),
            ],
            Grid::Threads,
            [w.cols, tokens.len(), 1],
        );
    }

    fn matmul(&self, w: &Weights, x: &Buf, out: &mut Buf) {
        let n = x.len / w.cols;
        assert!(x.len == n * w.cols && out.len == n * w.rows);
        // The kernels load 4 values (or whole 32-weight blocks) at a time.
        assert!(w.cols.is_multiple_of(4));
        let params = Params {
            rows: w.rows as u32,
            cols: w.cols as u32,
            ..Default::default()
        };
        // Batches go through the tiled kernel, which reads each weight once per 32 tokens.
        if n >= TILED_MIN_TOKENS {
            self.dispatch(
                &format!("matmul_tiled_{}", kernel_suffix(w.ty)),
                Params {
                    n: n as u32,
                    ..params
                },
                &[
                    (&w.buffer, w.offset),
                    (&x.buffer, x.offset),
                    (&out.buffer, out.offset),
                ],
                Grid::Groups(4 * SIMD_WIDTH),
                [w.rows.div_ceil(64), n.div_ceil(32), 1],
            );
            return;
        }
        self.dispatch(
            &format!("matmul_{}", kernel_suffix(w.ty)),
            params,
            &[
                (&w.buffer, w.offset),
                (&x.buffer, x.offset),
                (&out.buffer, out.offset),
            ],
            Grid::Groups(SIMD_WIDTH),
            [w.rows.div_ceil(matmul_rows(w.ty)), n, 1],
        );
    }

    fn rms_norm(&self, x: &Buf, weight: &Buf, eps: f32, out: &mut Buf) {
        let len = weight.len;
        assert!(x.len.is_multiple_of(len) && x.len == out.len);
        let params = Params {
            len: len as u32,
            eps,
            ..Default::default()
        };
        // The wide kernel wants whole simdgroups of 4-value threads in one threadgroup.
        let wide = len.is_multiple_of(4 * SIMD_WIDTH)
            && len / 4 <= self.pipelines["rms_norm"].maxTotalThreadsPerThreadgroup();
        let (kernel, threads) = if wide {
            ("rms_norm", len / 4)
        } else {
            ("rms_norm_any", SIMD_WIDTH)
        };
        self.dispatch(
            kernel,
            params,
            &[
                (&x.buffer, x.offset),
                (&weight.buffer, weight.offset),
                (&out.buffer, out.offset),
            ],
            Grid::Groups(threads),
            [x.len / len, 1, 1],
        );
    }

    fn rope(&self, x: &mut Buf, n: usize, head_dim: usize, pos: usize, freq_base: f32) {
        let heads = x.len / head_dim;
        let params = Params {
            head_dim: head_dim as u32,
            n_head: (heads / n) as u32,
            pos: pos as u32,
            ..Default::default()
        };
        let freqs = self.rope_frequencies(head_dim, freq_base);
        self.dispatch(
            "rope",
            params,
            &[(&x.buffer, x.offset), (&freqs, 0)],
            Grid::Threads,
            [head_dim / 2, heads, 1],
        );
    }

    fn attention(
        &self,
        shape: AttentionShape,
        q: &Buf,
        k: &Buf,
        v: &Buf,
        k_cache: &mut Buf,
        v_cache: &mut Buf,
        out: &mut Buf,
    ) {
        let AttentionShape {
            pos,
            n_head,
            n_head_kv,
            head_dim,
        } = shape;
        let step = n_head_kv * head_dim;
        let n = k.len / step;
        assert!(q.len == n * n_head * head_dim && out.len == q.len && k.len == v.len);
        assert!(
            head_dim.is_multiple_of(4),
            "the attention kernels load 4 values at a time"
        );
        assert!((pos + n) * step <= k_cache.len && k_cache.len == v_cache.len);

        for (src, cache) in [(k, &*k_cache), (v, &*v_cache)] {
            let params = Params {
                len: (pos * step) as u32,
                ..Default::default()
            };
            self.dispatch(
                "copy_at",
                params,
                &[(&src.buffer, src.offset), (&cache.buffer, cache.offset)],
                Grid::Threads,
                [src.len, 1, 1],
            );
        }

        let seen = pos + n; // positions visible to the last token of the batch
        let scores = self.alloc(n * n_head * seen);
        let params = Params {
            n: n as u32,
            head_dim: head_dim as u32,
            n_head: n_head as u32,
            n_head_kv: n_head_kv as u32,
            pos: pos as u32,
            len: seen as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            ..Default::default()
        };
        self.dispatch(
            "attention_scores",
            params,
            &[
                (&q.buffer, q.offset),
                (&k_cache.buffer, k_cache.offset),
                (&scores.buffer, 0),
            ],
            Grid::Threads,
            [seen, n_head, n],
        );
        self.dispatch(
            "attention_softmax",
            params,
            &[(&scores.buffer, 0)],
            Grid::Groups(SIMD_WIDTH),
            [n_head, n, 1],
        );
        self.dispatch(
            "attention_mix",
            params,
            &[
                (&scores.buffer, 0),
                (&v_cache.buffer, v_cache.offset),
                (&out.buffer, out.offset),
            ],
            Grid::Threads,
            [head_dim / 4, n_head, n],
        );
    }

    fn silu_mul(&self, gate: &Buf, up: &Buf, out: &mut Buf) {
        assert!(gate.len == up.len && gate.len == out.len);
        self.dispatch(
            "silu_mul",
            Params::default(),
            &[
                (&gate.buffer, gate.offset),
                (&up.buffer, up.offset),
                (&out.buffer, out.offset),
            ],
            Grid::Threads,
            [gate.len, 1, 1],
        );
    }

    fn add(&self, x: &mut Buf, y: &Buf) {
        assert!(x.len == y.len);
        self.dispatch(
            "add",
            Params::default(),
            &[(&x.buffer, x.offset), (&y.buffer, y.offset)],
            Grid::Threads,
            [x.len, 1, 1],
        );
    }
}
