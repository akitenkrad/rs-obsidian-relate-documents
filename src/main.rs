//! `obsidian-relate-documents` (crate `rs-obsidian-relate-documents`) —
//! compute `related-documents` Obsidian wiki-links for paper-report notes.
//!
//! Hybrid relatedness: Phase A (IDF-weighted tag/author cosine) widens recall;
//! Phase B (OpenAI embedding cosine) ranks every kept edge uniformly by
//! embedding cosine, with a B-veto that drops tag-only pairs whose embedding
//! cosine is below the floor. The result is merged into a symmetric,
//! degree-capped graph and the `related-documents:` YAML block of each report
//! is fully reconstructed (idempotent). DRY-RUN by default.

mod config;
mod corpus;
mod embed_cache;
mod frontmatter;
mod merge;
mod phase_a;
mod phase_b;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rusqlite::Connection;

use config::{Config, ENV_VAULT_ROOT};
use corpus::Paper;

/// Compute `related-documents` Obsidian wiki-links for paper-report notes.
#[derive(Parser, Debug)]
// Explicit `name` so `--version`/`--help` show the binary name
// (`obsidian-relate-documents`); clap otherwise falls back to the crate name.
#[command(name = "obsidian-relate-documents", version, about, long_about = None)]
struct Args {
    /// Write changes to disk (default: dry-run, no files touched).
    #[arg(long, default_value_t = false)]
    apply: bool,

    /// Embed only the first N papers (testing; reduces API cost).
    #[arg(long, value_name = "N")]
    limit_embed: Option<usize>,

    /// An A-only edge survives only if embedding cosine >= this floor.
    #[arg(long, default_value_t = 0.40, value_name = "F")]
    a_veto_cos: f64,

    /// Random seed for the 20-sample selection.
    #[arg(long, default_value_t = 42, value_name = "SEED")]
    seed: u64,

    /// Vault root on disk. Required: this flag or env OBSIDIAN_VAULT_ROOT.
    #[arg(long, value_name = "PATH")]
    vault_root: Option<String>,

    /// Reports directory, relative to the vault root.
    #[arg(long, value_name = "STR")]
    reports_rel: Option<String>,

    /// Report sub-folder to include (repeatable). Default: 00-General,
    /// 01-Simulation, 02-Security.
    #[arg(long, value_name = "NAME", action = clap::ArgAction::Append)]
    folders: Vec<String>,

    /// Protected relpath relative to <reports-rel> (repeatable): never
    /// written, but may still be a link target. Default: none.
    #[arg(long, value_name = "RELPATH", action = clap::ArgAction::Append)]
    protected: Vec<String>,

    /// Directory for the embedding cache + edge dump. Default: temp dir.
    #[arg(long, value_name = "DIR")]
    cache_dir: Option<String>,

    /// Cache DB path. Default: $HOME/.cache/obsidian-paper-cache/papers.db.
    #[arg(long, value_name = "PATH")]
    db_path: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cfg = build_config(&args)?;
    run(&args, &cfg)
}

/// Build the resolved [`Config`] from CLI args + environment.
///
/// Thin caller of [`Config::from_parts`]: all resolution rules (and their
/// error cases) live there so they stay unit-testable without process/env.
fn build_config(args: &Args) -> Result<Config> {
    Config::from_parts(
        args.vault_root.clone(),
        std::env::var(ENV_VAULT_ROOT).ok(),
        args.reports_rel.clone(),
        args.folders.clone(),
        args.protected.clone(),
        args.cache_dir.clone(),
        std::env::temp_dir(),
        args.db_path.clone(),
        std::env::var("HOME").ok(),
    )
}

/// Round to 4 decimals for percentile display.
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// numpy-compatible linear-interpolation percentiles.
fn percentiles(values: &[f64], ps: &[u8]) -> Vec<Option<f64>> {
    if values.is_empty() {
        return ps.iter().map(|_| None).collect();
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ps.iter()
        .map(|&p| {
            let n = v.len();
            if n == 1 {
                return Some(round4(v[0]));
            }
            // numpy default ('linear'): pos = p/100 * (n-1).
            let pos = (p as f64) / 100.0 * ((n - 1) as f64);
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            let frac = pos - lo as f64;
            let val = v[lo] + (v[hi] - v[lo]) * frac;
            Some(round4(val))
        })
        .collect()
}

/// Format an `Option<f64>` percentile like Python's `None`/number.
fn pf(o: Option<f64>) -> String {
    match o {
        Some(x) => format!("{x}"),
        None => "None".to_string(),
    }
}

#[allow(clippy::too_many_lines)]
fn run(args: &Args, cfg: &Config) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    cfg.ensure_cache_dir()?;

    let db = &cfg.db_path;
    let conn = Connection::open(db).with_context(|| format!("open cache DB {}", db.display()))?;
    let papers: Vec<Paper> = corpus::load_corpus(&conn, cfg)?;
    let n = papers.len();
    let targets_map = corpus::build_target_resolver(&papers, cfg);

    let protected_set: HashSet<&str> = cfg.protected_relpaths.iter().map(String::as_str).collect();
    let protected_idx: HashSet<usize> = papers
        .iter()
        .enumerate()
        .filter(|(_, p)| protected_set.contains(p.relpath.as_str()))
        .map(|(i, _)| i)
        .collect();

    let bar = "=".repeat(72);
    println!("{bar}");
    println!("{}", if args.apply { "[APPLY]" } else { "[DRY-RUN]" });
    println!("Corpus size            : {n}");
    println!("Protected (skip-write) : {}", protected_idx.len());
    println!("{bar}");

    // ---- Phase A ----
    let a = phase_a::phase_a(&papers);
    println!("\n--- Phase A (IDF-weighted tag/author cosine) ---");
    println!(
        "Distinct terms              : {} (nonzero idf: {})",
        a.n_terms, a.n_terms_nonzero_idf
    );
    println!("Papers with >=1 A-edge      : {}", a.papers_with_edge);
    println!("Total A-edges (undirected)  : {}", a.total_edges);
    let ap = percentiles(&a.weights, &[50, 75, 90, 95, 99]);
    println!(
        "A edge-weight percentiles   : p50={} p75={} p90={} p95={} p99={}",
        pf(ap[0]),
        pf(ap[1]),
        pf(ap[2]),
        pf(ap[3]),
        pf(ap[4])
    );

    // ---- Phase B ----
    let b = phase_b::phase_b(&papers, args.limit_embed, cfg)?;
    println!("\n--- Phase B (OpenAI embedding cosine) ---");
    println!(
        "Embedding batches/API calls : {} / {} (cache hits: {})",
        b.batches, b.api_calls, b.cache_hits
    );
    println!("Papers embedded             : {}", b.embedded);
    println!("Papers with >=1 B-edge      : {}", b.papers_with_edge);
    println!("Total B-edges (undirected)  : {}", b.total_edges);
    let bcos: Vec<f64> = b.cos_values.iter().map(|&x| x as f64).collect();
    let bp = percentiles(&bcos, &[50, 75, 90, 95, 99]);
    println!(
        "B cosine percentiles        : p50={} p75={} p90={} p95={} p99={}",
        pf(bp[0]),
        pf(bp[1]),
        pf(bp[2]),
        pf(bp[3]),
        pf(bp[4])
    );
    println!("API tokens (reported)       : {}", b.total_tokens);

    // ---- Merge ----
    let edge_dump_path = cfg.edge_dump_path();
    let (final_g, dec) = merge::merge_graph(
        &papers,
        &a.a_edges,
        &b.b_edges,
        &b.m,
        &b.have_vec,
        args.a_veto_cos,
        &edge_dump_path,
    )?;

    println!(
        "\n--- Merge edge decisions (B-veto, floor={}) ---",
        args.a_veto_cos
    );
    println!("  keep (B/AB)         : {}", dec.keep_b_ab);
    println!("  keep (A-veto pass)  : {}", dec.keep_a_veto_pass);
    println!("  veto-A (tag-only)   : {}", dec.veto_a);
    println!("  below-B (B cand)    : {}", dec.below_b);

    // Protected files are never writers (but may remain targets elsewhere).
    let writer_final: HashMap<usize, Vec<(usize, f32, String)>> = final_g
        .iter()
        .filter(|(i, _)| !protected_idx.contains(i))
        .map(|(i, v)| (*i, v.clone()))
        .collect();

    // ---- Final stats ----
    let mut deg_hist = [0usize; merge::HARD_CAP + 1];
    let mut papers_with_link = 0usize;
    for i in 0..n {
        if protected_idx.contains(&i) {
            continue;
        }
        let d = writer_final.get(&i).map_or(0, Vec::len);
        let d = d.min(merge::HARD_CAP);
        deg_hist[d] += 1;
        if d > 0 {
            papers_with_link += 1;
        }
    }
    let writable = n - protected_idx.len();
    let empty = writable - papers_with_link;

    println!("\n--- Final merged graph (writers only; protected excluded) ---");
    println!("Writable reports            : {writable}");
    println!("Will get >=1 link           : {papers_with_link}");
    println!("Stay empty (related: [])    : {empty}");
    println!("Link-count histogram (0..10):");
    for (k, &count) in deg_hist.iter().enumerate() {
        let hbar = "#".repeat((count / 10).min(60));
        println!("  {k:2}: {count:5} {hbar}");
    }

    // Reciprocity over the FULL merged graph (before protected-writer trim).
    let mut asym = 0usize;
    for (i, lst) in &final_g {
        for (j, _w, _s) in lst {
            let reciprocated = final_g
                .get(j)
                .map(|p| p.iter().any(|(x, _, _)| x == i))
                .unwrap_or(false);
            if !reciprocated {
                asym += 1;
            }
        }
    }
    println!("Asymmetric edges (must be 0): {asym}");

    // Dead-target + self-link checks over what would be WRITTEN.
    let mut dead = 0usize;
    let mut self_links = 0usize;
    for (i, lst) in &writer_final {
        for (j, _w, _s) in lst {
            if j == i {
                self_links += 1;
            }
            if *j >= n || !targets_map.contains_key(j) {
                dead += 1;
            }
        }
    }
    println!("Dead targets (must be 0)    : {dead}");
    println!("Self-links (must be 0)      : {self_links}");

    println!("\nTOTAL embedding API calls   : {}", b.api_calls);

    // ---- 20 random samples ----
    println!("\n{bar}");
    println!("20 RANDOM SAMPLE PAPERS (title -> related [target | weight | src])");
    println!("{bar}");

    let mut linked_writers: Vec<usize> = writer_final
        .iter()
        .filter(|(_, v)| !v.is_empty())
        .map(|(i, _)| *i)
        .collect();
    linked_writers.sort_unstable();

    let a_driven: Vec<usize> = linked_writers
        .iter()
        .copied()
        .filter(|i| {
            writer_final[i]
                .iter()
                .any(|(_, _, s)| s == "A" || s == "AB")
        })
        .collect();
    let b_driven: Vec<usize> = linked_writers
        .iter()
        .copied()
        .filter(|i| writer_final[i].iter().any(|(_, _, s)| s == "B"))
        .collect();

    let mut chosen: Vec<usize> = Vec::new();
    let mut pool: Vec<usize> = linked_writers.clone();
    pool.shuffle(&mut rng);

    fn take(src_list: &[usize], k: usize, chosen: &mut Vec<usize>, rng: &mut ChaCha8Rng) {
        let mut local: Vec<usize> = src_list
            .iter()
            .copied()
            .filter(|x| !chosen.contains(x))
            .collect();
        local.shuffle(rng);
        for x in local.into_iter().take(k) {
            chosen.push(x);
        }
    }

    take(&a_driven, 8, &mut chosen, &mut rng);
    take(&b_driven, 8, &mut chosen, &mut rng);
    let rest = 20usize.saturating_sub(chosen.len());
    take(&pool, rest, &mut chosen, &mut rng);
    chosen.truncate(20);

    for &i in &chosen {
        let p = &papers[i];
        println!("\n[{}] {}", p.folder, trunc(&p.title, 90));
        for (j, w, src) in &writer_final[&i] {
            let tp = &papers[*j];
            println!("   ({src:2} {w:.3})  {}", trunc(&tp.title, 80));
        }
    }

    // ---- Apply ----
    if args.apply {
        println!("\n[APPLY] Writing files ...");
        let mut written = 0usize;
        let mut skipped_protected = 0usize;
        let mut skipped_missing = 0usize;
        for (i, p) in papers.iter().enumerate() {
            if protected_idx.contains(&i) {
                skipped_protected += 1;
                continue;
            }
            if !PathBuf::from(&p.file_path).exists() {
                skipped_missing += 1;
                continue;
            }
            let tgts: Vec<String> = writer_final
                .get(&i)
                .map(|v| v.iter().map(|(j, _, _)| targets_map[j].clone()).collect())
                .unwrap_or_default();
            let changed = frontmatter::reconstruct_file(Path::new(&p.file_path), &tgts)?;
            if changed {
                written += 1;
            }
        }
        println!(
            "[APPLY] files changed: {written}, protected skipped: \
             {skipped_protected}, missing skipped: {skipped_missing}"
        );
    } else {
        println!("\n[DRY-RUN] No files written. Re-run with --apply to write.");
    }

    Ok(())
}

/// Truncate a string to `max` chars for display (char-safe).
fn trunc(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_match_numpy_linear() {
        // numpy.percentile([1..=5], [50,75]) == [3.0, 4.0].
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = percentiles(&v, &[50, 75, 90, 95, 99]);
        assert_eq!(p[0], Some(3.0));
        assert_eq!(p[1], Some(4.0));
        // p90 of 1..5 = 4.6, p95=4.8, p99=4.96.
        assert_eq!(p[2], Some(4.6));
        assert_eq!(p[3], Some(4.8));
        assert_eq!(p[4], Some(4.96));
    }

    #[test]
    fn percentiles_empty_is_none() {
        let p = percentiles(&[], &[50, 99]);
        assert_eq!(p, vec![None, None]);
    }

    #[test]
    fn percentiles_single_value() {
        let p = percentiles(&[7.0], &[50, 99]);
        assert_eq!(p, vec![Some(7.0), Some(7.0)]);
    }
}
