//! Next-token sampling: greedy, temperature, top-k, top-p, min-p, with a seeded RNG.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// 0 = greedy (argmax).
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens. 0 = off.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to `top_p`. 1 = off.
    pub top_p: f32,
    /// Drop tokens less likely than `min_p` times the most likely one. 0 = off.
    pub min_p: f32,
    pub seed: u64,
}

pub struct Sampler {
    config: SamplerConfig,
    rng: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Sampler {
            config,
            rng: config.seed,
        }
    }

    /// Filters are applied in Hugging Face's order: temperature, top-k, top-p, then min-p.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let c = self.config;
        if c.temperature <= 0.0 {
            return argmax(logits);
        }

        // Candidates sorted by logit, highest first (ties: lowest id first).
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        ids.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
        if c.top_k > 0 {
            ids.truncate(c.top_k);
        }

        // Softmax with temperature, subtracting the max first.
        let max = logits[ids[0] as usize];
        let mut probs: Vec<f32> = ids
            .iter()
            .map(|id| ((logits[*id as usize] - max) / c.temperature).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);

        if c.top_p < 1.0 {
            let mut cumulative = 0.0;
            let keep = probs
                .iter()
                .position(|p| {
                    cumulative += p;
                    cumulative >= c.top_p
                })
                .map_or(probs.len(), |i| i + 1);
            probs.truncate(keep);
        }
        if c.min_p > 0.0 {
            let threshold = c.min_p * probs[0];
            let keep = probs.iter().take_while(|p| **p >= threshold).count();
            probs.truncate(keep.max(1));
        }

        // Draw from what is left (no need to renormalize: scale the draw instead).
        let total: f32 = probs.iter().sum();
        let mut draw = self.uniform() * total;
        for (id, p) in ids.iter().zip(&probs) {
            draw -= p;
            if draw < 0.0 {
                return *id;
            }
        }
        ids[probs.len() - 1] // rounding left a sliver: take the last candidate
    }

    /// Uniform in [0, 1), from splitmix64.
    fn uniform(&mut self) -> f32 {
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Index of the highest logit; the first one on ties, like `torch.argmax`.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: SamplerConfig = SamplerConfig {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 42,
    };

    fn draws(config: SamplerConfig, logits: &[f32], n: usize) -> Vec<u32> {
        let mut s = Sampler::new(config);
        (0..n).map(|_| s.sample(logits)).collect()
    }

    /// ln of the probabilities 0.1, 0.6, 0.3
    fn logits() -> Vec<f32> {
        [0.1f32, 0.6, 0.3].iter().map(|p| p.ln()).collect()
    }

    #[test]
    fn greedy_is_first_argmax() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        let greedy = SamplerConfig {
            temperature: 0.0,
            ..BASE
        };
        assert!(draws(greedy, &logits(), 20).iter().all(|t| *t == 1));
    }

    #[test]
    fn follows_the_distribution() {
        let n = 20_000;
        let d = draws(BASE, &logits(), n);
        for (id, p) in [(0u32, 0.1f32), (1, 0.6), (2, 0.3)] {
            let freq = d.iter().filter(|t| **t == id).count() as f32 / n as f32;
            assert!((freq - p).abs() < 0.015, "token {id}: {freq} vs {p}");
        }
    }

    #[test]
    fn temperature_sharpens() {
        // p^(1/T) renormalized, T = 0.5: 0.01, 0.36, 0.09 over 0.46
        let cold = SamplerConfig {
            temperature: 0.5,
            ..BASE
        };
        let n = 20_000;
        let d = draws(cold, &logits(), n);
        let freq = d.iter().filter(|t| **t == 1).count() as f32 / n as f32;
        assert!((freq - 0.36 / 0.46).abs() < 0.015, "{freq}");
    }

    #[test]
    fn filters_restrict_candidates() {
        let l = logits();
        let only = |config, allowed: &[u32]| {
            let d = draws(config, &l, 2000);
            assert!(d.iter().all(|t| allowed.contains(t)), "{config:?}");
            // every allowed token does show up
            assert!(allowed.iter().all(|a| d.contains(a)), "{config:?}");
        };
        only(SamplerConfig { top_k: 1, ..BASE }, &[1]);
        only(SamplerConfig { top_k: 2, ..BASE }, &[1, 2]);
        // 0.6 alone is below 0.7, 0.6 + 0.3 reaches it
        only(SamplerConfig { top_p: 0.7, ..BASE }, &[1, 2]);
        only(SamplerConfig { top_p: 0.5, ..BASE }, &[1]);
        // threshold 0.4 * 0.6 = 0.24 keeps 0.6 and 0.3
        only(SamplerConfig { min_p: 0.4, ..BASE }, &[1, 2]);
        only(SamplerConfig { min_p: 1.0, ..BASE }, &[1]);
    }

    #[test]
    fn seed_makes_it_reproducible() {
        let l = logits();
        assert_eq!(draws(BASE, &l, 50), draws(BASE, &l, 50));
        let other = SamplerConfig { seed: 43, ..BASE };
        assert_ne!(draws(BASE, &l, 50), draws(other, &l, 50));
    }
}
