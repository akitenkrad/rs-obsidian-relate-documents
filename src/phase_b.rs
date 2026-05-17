//! Phase B: OpenAI embedding (`text-embedding-3-small`) cosine over
//! title + 要約 body.
//!
//! Numeric work is `f32` (matching the Python `numpy.float32` embedding path).

use std::collections::HashMap;
use std::io::Read;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use serde::Deserialize;

use crate::config::Config;
use crate::corpus::Paper;
use crate::embed_cache::{self, EmbedCache};

/// Embedding model.
pub const EMBED_MODEL: &str = "text-embedding-3-small";
/// Embeddings endpoint.
pub const EMBED_URL: &str = "https://api.openai.com/v1/embeddings";
/// Batch size per API call.
pub const EMBED_BATCH: usize = 96;
/// Embedding dimensionality.
pub const EMBED_DIM: usize = 1536;
/// Minimum cosine for a B edge.
pub const B_COSINE_MIN: f32 = 0.55;
/// Per-node top-K B edges kept.
pub const B_TOPK: usize = 8;

/// Lazily-built regexes (compiled once).
struct ReSet {
    summary_heading: Regex,
    next_h2: Regex,
    html_comment: Regex,
    callout_hdr: Regex,
    embed_link: Regex,
    ws: Regex,
    blockquote: Regex,
}

impl ReSet {
    /// Shared, compile-once regex set.
    fn shared() -> &'static ReSet {
        use std::sync::OnceLock;
        static SET: OnceLock<ReSet> = OnceLock::new();
        SET.get_or_init(ReSet::new)
    }

    fn new() -> Self {
        ReSet {
            // ^##\s*(?:2\.\s*)?要約\s*$
            summary_heading: Regex::new(r"^##\s*(?:2\.\s*)?要約\s*$").unwrap(),
            next_h2: Regex::new(r"^##\s").unwrap(),
            // <!--.*?--> with DOTALL.
            html_comment: Regex::new(r"(?s)<!--.*?-->").unwrap(),
            callout_hdr: Regex::new(r"^\s*>\s*\[!.*?\]").unwrap(),
            embed_link: Regex::new(r"!?\[\[.*?\]\]").unwrap(),
            ws: Regex::new(r"\s+").unwrap(),
            blockquote: Regex::new(r"^\s*>\s?").unwrap(),
        }
    }
}

/// Extract the title-agnostic body text (要約 section, or fallback).
///
/// Mirrors the Python `extract_embed_text` exactly:
/// 1. Skip frontmatter (line0 trimmed `---` … next trimmed `---`).
/// 2. First line matching `^##\s*(?:2\.\s*)?要約\s*$` → collect until the
///    next line matching `^##\s`; else fallback = whole body.
/// 3. Remove HTML comments (DOTALL), drop callout-header lines, strip a
///    leading blockquote `>`, replace `[[..]]`/`![[..]]` with a space.
/// 4. Collapse whitespace runs to single space, trim.
/// 5. If fallback, truncate to 1500 chars (char-safe).
pub fn extract_embed_text(file_path: &str) -> String {
    let content = match read_utf8(file_path) {
        Ok(c) => c,
        Err(_) => return String::new(),
    };
    extract_embed_text_from(&content, ReSet::shared())
}

/// Read a file as UTF-8 (lossless; returns Err on IO failure).
fn read_utf8(path: &str) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Core of [`extract_embed_text`] operating on already-loaded content
/// (separated so it is unit-testable without touching the filesystem).
fn extract_embed_text_from(content: &str, re: &ReSet) -> String {
    let lines: Vec<&str> = content.split('\n').collect();

    // Skip frontmatter.
    let mut body_start = 0usize;
    if lines.first().map(|l| l.trim()) == Some("---") {
        for (idx, ln) in lines.iter().enumerate().skip(1) {
            if ln.trim() == "---" {
                body_start = idx + 1;
                break;
            }
        }
    }

    // Find the 要約 section.
    let mut section_lines: Option<Vec<&str>> = None;
    for idx in body_start..lines.len() {
        if re.summary_heading.is_match(lines[idx]) {
            let mut collected: Vec<&str> = Vec::new();
            for &ln in &lines[idx + 1..] {
                if re.next_h2.is_match(ln) {
                    break;
                }
                collected.push(ln);
            }
            section_lines = Some(collected);
            break;
        }
    }

    let (section, fallback) = match section_lines {
        Some(s) => (s, false),
        None => (lines[body_start..].to_vec(), true),
    };

    let text = section.join("\n");
    let text = re.html_comment.replace_all(&text, " ");

    let mut cleaned: Vec<String> = Vec::new();
    for ln in text.split('\n') {
        if re.callout_hdr.is_match(ln) {
            continue;
        }
        let ln = re.blockquote.replace(ln, "");
        let ln = re.embed_link.replace_all(&ln, " ");
        cleaned.push(ln.into_owned());
    }
    let joined = cleaned.join(" ");
    let out = re.ws.replace_all(&joined, " ");
    let out = out.trim().to_string();

    if fallback {
        truncate_chars(&out, 1500)
    } else {
        out
    }
}

/// Char-safe truncation (Python string slicing is by code point, not byte).
pub fn truncate_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// Build the embed input string for every paper:
/// `(title + "\n" + body).trim()`, then truncate to 2000 chars (char-safe).
pub fn build_embed_inputs(papers: &[Paper]) -> Vec<String> {
    papers
        .iter()
        .map(|p| {
            let body = extract_embed_text(&p.file_path);
            let combined = format!("{}\n{}", p.title, body);
            let combined = combined.trim();
            truncate_chars(combined, 2000)
        })
        .collect()
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Deserialize, Default)]
struct Usage {
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
    #[serde(default)]
    usage: Usage,
}

/// One OpenAI embeddings call with retry/backoff.
///
/// Retries on HTTP 429 or >=500 with exponential backoff starting at 2s,
/// doubling, max 5 attempts. Any other non-200 → error.
fn openai_embed(
    client: &reqwest::blocking::Client,
    batch: &[String],
    api_key: &str,
) -> Result<(Vec<EmbeddingItem>, u64)> {
    let payload = serde_json::json!({ "model": EMBED_MODEL, "input": batch });
    let mut delay = Duration::from_secs(2);
    for attempt in 0..5 {
        let resp = client
            .post(EMBED_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&payload)
            .timeout(Duration::from_secs(120))
            .send()
            .context("send embeddings request")?;
        let status = resp.status();
        if status.as_u16() == 200 {
            let parsed: EmbeddingResponse = resp.json().context("parse embeddings response")?;
            return Ok((parsed.data, parsed.usage.total_tokens));
        }
        if status.as_u16() == 429 || status.as_u16() >= 500 {
            if attempt == 4 {
                let body = resp.text().unwrap_or_default();
                bail!("embeddings HTTP {status} after retries: {body}");
            }
            std::thread::sleep(delay);
            delay *= 2;
            continue;
        }
        let body = resp.text().unwrap_or_default();
        bail!("embeddings HTTP {status}: {body}");
    }
    Err(anyhow!("embedding retries exhausted"))
}

/// Phase-B result.
pub struct PhaseBResult {
    /// `b_edges[i] -> {j: cosine}` (per-node top-8, cosine >= 0.55).
    pub b_edges: HashMap<usize, HashMap<usize, f32>>,
    /// Embedding matrix, row-major `n x EMBED_DIM`, L2-normalized rows.
    pub m: Vec<Vec<f32>>,
    /// Whether row `i` has a valid vector.
    pub have_vec: Vec<bool>,
    /// Number of API batches issued.
    pub batches: usize,
    /// Number of API calls issued (== batches).
    pub api_calls: usize,
    /// Reported total tokens.
    pub total_tokens: u64,
    /// Number of papers with a vector.
    pub embedded: usize,
    /// Number of cache hits.
    pub cache_hits: usize,
    /// Number of papers with >=1 B-edge.
    pub papers_with_edge: usize,
    /// Total undirected B-edges.
    pub total_edges: usize,
    /// Undirected cosine values for percentile stats.
    pub cos_values: Vec<f32>,
}

/// Compute Phase B. `limit_embed` restricts embedding to the first N papers
/// (testing). Reads `OPENAI_API_KEY` from the environment. The content-hash
/// embedding cache lives at `cfg.embed_cache_path()`.
pub fn phase_b(papers: &[Paper], limit_embed: Option<usize>, cfg: &Config) -> Result<PhaseBResult> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| anyhow!("OPENAI_API_KEY not set"))?;

    let n = papers.len();
    let embed_idx: Vec<usize> = match limit_embed {
        Some(k) => (0..n).take(k).collect(),
        None => (0..n).collect(),
    };

    let inputs = build_embed_inputs(papers);

    let mut m: Vec<Vec<f32>> = vec![vec![0.0_f32; EMBED_DIM]; n];
    let mut have_vec = vec![false; n];

    // Content-hash cache.
    let cache_path = cfg.embed_cache_path();
    let mut cache: EmbedCache = embed_cache::load(&cache_path);
    let mut cache_hits = 0usize;
    let mut hash_of: HashMap<usize, String> = HashMap::new();
    for &i in &embed_idx {
        let h = embed_cache::hash_input(&inputs[i]);
        if let Some(v) = cache.get(&h) {
            m[i] = v.clone();
            have_vec[i] = true;
            cache_hits += 1;
        }
        hash_of.insert(i, h);
    }

    // Embed the misses.
    let missing: Vec<(usize, String)> = embed_idx
        .iter()
        .filter(|&&i| !have_vec[i])
        .map(|&i| (i, inputs[i].clone()))
        .collect();

    let client = reqwest::blocking::Client::new();
    let mut batches = 0usize;
    let mut api_calls = 0usize;
    let mut total_tokens = 0u64;

    let mut start = 0usize;
    while start < missing.len() {
        let end = (start + EMBED_BATCH).min(missing.len());
        let chunk = &missing[start..end];
        let idxs: Vec<usize> = chunk.iter().map(|c| c.0).collect();
        let texts: Vec<String> = chunk
            .iter()
            .map(|c| {
                if c.1.is_empty() {
                    " ".to_string()
                } else {
                    c.1.clone()
                }
            })
            .collect();
        let (data, tokens) = openai_embed(&client, &texts, &api_key)?;
        api_calls += 1;
        batches += 1;
        total_tokens += tokens;
        for item in data {
            // Map back by the returned position within the batch.
            let row = idxs[item.index];
            let v = l2_normalize(item.embedding);
            if let Some(h) = hash_of.get(&row) {
                cache.insert(h.clone(), v.clone());
            }
            m[row] = v;
            have_vec[row] = true;
        }
        start = end;
    }
    embed_cache::save(&cache_path, &cache)?;

    // B edges: per node, top-8 neighbors with cosine >= 0.55.
    let valid: Vec<usize> = (0..n).filter(|&i| have_vec[i]).collect();
    let mut b_edges: HashMap<usize, HashMap<usize, f32>> = HashMap::new();
    let mut cos_values: Vec<f32> = Vec::new();

    if valid.len() >= 2 {
        for (a_pos, &i) in valid.iter().enumerate() {
            // Compute cosine to every other valid node.
            let mut row: Vec<(usize, f32)> = Vec::with_capacity(valid.len());
            for (b_pos, &j) in valid.iter().enumerate() {
                let c = if b_pos == a_pos {
                    -1.0_f32
                } else {
                    dot(&m[i], &m[j])
                };
                row.push((b_pos, c));
            }
            // argsort(-row): descending by cosine. numpy argsort is stable;
            // use a stable sort with original index as the implicit tiebreak.
            row.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
            // Keep up to B_TOPK neighbors, stopping at the first cosine
            // below the floor (rows are descending, so this is exact).
            for &(b_pos, c) in row
                .iter()
                .take(B_TOPK)
                .take_while(|&&(_, c)| c >= B_COSINE_MIN)
            {
                let j = valid[b_pos];
                b_edges.entry(i).or_default().insert(j, c);
            }
        }
        // Symmetric cosine stats.
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        for (&i, nbrs) in &b_edges {
            for (&j, &c) in nbrs {
                let key = if i < j { (i, j) } else { (j, i) };
                if seen.insert(key) {
                    cos_values.push(c);
                }
            }
        }
    }

    let embedded = have_vec.iter().filter(|&&b| b).count();
    let papers_with_edge = b_edges.values().filter(|m| !m.is_empty()).count();

    Ok(PhaseBResult {
        b_edges,
        m,
        have_vec,
        batches,
        api_calls,
        total_tokens,
        embedded,
        cache_hits,
        papers_with_edge,
        total_edges: cos_values.len(),
        cos_values,
    })
}

/// L2-normalize a vector in place (skip if norm is 0, matching Python).
pub fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// f32 dot product of two equal-length vectors.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn re() -> ReSet {
        ReSet::new()
    }

    #[test]
    fn extracts_summary_section_and_strips_noise() {
        let content = "\
---
title: x
---
# Heading

## 2. 要約
これは要約です。
> [!note] callout title
> 引用された行のテキスト
ここに ![[embedded.png]] と [[link]] がある。
<!-- this is a
multiline comment -->
本文の続き。

## 次のセクション
無視される本文。
";
        let out = extract_embed_text_from(content, &re());
        // Summary content kept; next ## section excluded.
        assert!(out.contains("これは要約です。"));
        assert!(out.contains("本文の続き。"));
        assert!(!out.contains("無視される"));
        // Callout header line dropped, blockquote marker stripped.
        assert!(!out.contains("[!note]"));
        assert!(out.contains("引用された行のテキスト"));
        // Embeds replaced with space.
        assert!(!out.contains("[[link]]"));
        assert!(!out.contains("embedded.png"));
        // HTML comment removed.
        assert!(!out.contains("multiline comment"));
        // Whitespace collapsed.
        assert!(!out.contains("  "));
    }

    #[test]
    fn fallback_when_no_summary_truncates_to_1500() {
        let mut body = String::from("---\nt: 1\n---\n");
        body.push_str(&"あ".repeat(3000));
        let out = extract_embed_text_from(&body, &re());
        // Fallback path: char-safe truncation to 1500 chars.
        assert_eq!(out.chars().count(), 1500);
        assert!(out.chars().all(|c| c == 'あ'));
    }

    #[test]
    fn no_frontmatter_treated_as_body() {
        let content = "## 要約\nプレーンな本文。\n";
        let out = extract_embed_text_from(content, &re());
        assert_eq!(out, "プレーンな本文。");
    }

    #[test]
    fn truncate_chars_is_codepoint_safe_on_multibyte() {
        // 5 multibyte chars (each 3 bytes in UTF-8); byte-slicing at 2000
        // would panic mid-codepoint — char truncation must not.
        let s = "日本語日本語日本語";
        let t = truncate_chars(s, 4);
        assert_eq!(t, "日本語日");
        assert_eq!(t.chars().count(), 4);
        // No truncation when under the limit.
        assert_eq!(truncate_chars(s, 100), s);
    }

    #[test]
    fn summary_heading_variants_match() {
        let r = re();
        assert!(r.summary_heading.is_match("## 要約"));
        assert!(r.summary_heading.is_match("##要約"));
        assert!(r.summary_heading.is_match("## 2. 要約"));
        assert!(!r.summary_heading.is_match("## 要約の詳細"));
    }

    #[test]
    fn l2_normalize_unit_and_zero() {
        let v = l2_normalize(vec![3.0, 4.0]);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
        // Zero vector left unchanged (no division by zero).
        assert_eq!(l2_normalize(vec![0.0, 0.0]), vec![0.0, 0.0]);
    }
}
