//! Merge Phase A + Phase B into a symmetric, degree-capped graph.
//!
//! Ranking weight for EVERY kept edge is the embedding cosine (`ec`),
//! uniformly — A's tag cosine never enters ranking. Phase A only widens
//! recall. Decision table per candidate pair `(i, j)`:
//!
//! | condition                              | decision | src        |
//! |----------------------------------------|----------|------------|
//! | no vector on either endpoint           | drop     | —          |
//! | `ec >= 0.55`                           | keep     | `AB`/`B`   |
//! | A-edge & `ec >= a_veto_cos`            | keep     | `A`        |
//! | A-edge & `ec <  a_veto_cos`            | veto-A   | —          |
//! | else (B candidate, `ec < 0.55`)        | below-B  | —          |

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::corpus::Paper;
use crate::phase_b::dot;

/// `ec >= this` keeps a pair outright (also Phase B's own threshold).
pub const B_COSINE_MIN: f32 = 0.55;
/// Per-node top incident edges before reciprocity.
pub const MERGE_TOPK: usize = 8;
/// Absolute max degree per node.
pub const HARD_CAP: usize = 10;

/// One row of the edge-decision dump (`<cache-dir>/relate_edges_rs.json`).
#[derive(Serialize)]
struct EdgeDumpRow {
    i: usize,
    j: usize,
    a_cos: Option<f64>,
    emb_cos: Option<f64>,
    is_a: bool,
    is_b: bool,
    decision: String,
    src: Option<String>,
}

/// Round to 4 decimals (Python `round(x, 4)`, banker's rounding is not used
/// by the analysis so plain round-half-away is fine and matches the dump's
/// purpose; values are display-only).
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Result of [`merge_graph`]: `i -> [(j, weight, src)]`, weight-descending.
pub type MergedGraph = HashMap<usize, Vec<(usize, f32, String)>>;

/// Edge-decision tallies for the printed breakdown.
#[derive(Default)]
pub struct EdgeDecisionStats {
    /// `keep` with src `B` or `AB`.
    pub keep_b_ab: usize,
    /// `keep` with src `A` (A-veto pass).
    pub keep_a_veto_pass: usize,
    /// `veto-A` (tag-only, embedding too low).
    pub veto_a: usize,
    /// `below-B` (B candidate, embedding < 0.55).
    pub below_b: usize,
}

/// Merge A/B edges. `m` is the row-major embedding matrix; `have_vec[i]`
/// indicates a valid row. `a_veto_cos` is the A-only survival floor.
///
/// Also writes the edge-decision dump JSON to `edge_dump_path`.
#[allow(clippy::too_many_arguments)]
pub fn merge_graph(
    papers: &[Paper],
    a_edges: &HashMap<usize, HashMap<usize, f64>>,
    b_edges: &HashMap<usize, HashMap<usize, f32>>,
    m: &[Vec<f32>],
    have_vec: &[bool],
    a_veto_cos: f64,
    edge_dump_path: &std::path::Path,
) -> anyhow::Result<(MergedGraph, EdgeDecisionStats)> {
    let _ = papers;

    let emb_cos = |i: usize, j: usize| -> Option<f32> {
        if have_vec[i] && have_vec[j] {
            Some(dot(&m[i], &m[j]))
        } else {
            None
        }
    };

    // Collect every candidate unordered pair from A or B.
    let mut a_pairs: HashMap<(usize, usize), f64> = HashMap::new();
    for (&i, nbrs) in a_edges {
        for (&j, &c) in nbrs {
            let key = if i < j { (i, j) } else { (j, i) };
            let e = a_pairs.entry(key).or_insert(0.0);
            if c > *e {
                *e = c;
            }
        }
    }
    let mut b_pairs: HashSet<(usize, usize)> = HashSet::new();
    for (&i, nbrs) in b_edges {
        for &j in nbrs.keys() {
            b_pairs.insert(if i < j { (i, j) } else { (j, i) });
        }
    }
    let mut all_pairs: Vec<(usize, usize)> = a_pairs
        .keys()
        .copied()
        .chain(b_pairs.iter().copied())
        .collect();
    all_pairs.sort_unstable();
    all_pairs.dedup();

    let mut pair_w: HashMap<(usize, usize), (f32, String)> = HashMap::new();
    let mut edge_dump: Vec<EdgeDumpRow> = Vec::with_capacity(all_pairs.len());
    let mut stats = EdgeDecisionStats::default();

    for key in &all_pairs {
        let (i, j) = *key;
        let is_a = a_pairs.contains_key(key);
        let is_b = b_pairs.contains(key);
        let ec = emb_cos(i, j);
        let a_cos = a_pairs.get(key).copied();

        let (decision, src): (&str, Option<&str>) = match ec {
            None => ("drop-no-vec", None),
            Some(e) if (e as f64) >= B_COSINE_MIN as f64 => {
                ("keep", Some(if is_a { "AB" } else { "B" }))
            }
            Some(e) if is_a && (e as f64) >= a_veto_cos => ("keep", Some("A")),
            Some(_) if is_a => ("veto-A", None),
            Some(_) => ("below-B", None),
        };

        match (decision, src) {
            ("keep", Some("A")) => stats.keep_a_veto_pass += 1,
            ("keep", Some(_)) => stats.keep_b_ab += 1,
            ("veto-A", _) => stats.veto_a += 1,
            ("below-B", _) => stats.below_b += 1,
            _ => {}
        }

        if let Some(s) = src {
            // Ranking weight is the embedding cosine, uniformly.
            pair_w.insert(*key, (ec.unwrap(), s.to_string()));
        }

        edge_dump.push(EdgeDumpRow {
            i,
            j,
            a_cos: a_cos.map(round4),
            emb_cos: ec.map(|e| round4(e as f64)),
            is_a,
            is_b,
            decision: decision.to_string(),
            src: src.map(|s| s.to_string()),
        });
    }

    // Best-effort dump (Python swallows OSError).
    if let Ok(json) = serde_json::to_string(&edge_dump) {
        let _ = std::fs::write(edge_dump_path, json);
    }

    // Incident edges per node.
    let mut incident: HashMap<usize, Vec<(f32, usize, String)>> = HashMap::new();
    for (&(i, j), (w, src)) in &pair_w {
        incident.entry(i).or_default().push((*w, j, src.clone()));
        incident.entry(j).or_default().push((*w, i, src.clone()));
    }

    // Step 1: each node keeps its top MERGE_TOPK incident edges by weight.
    // A pair is kept if kept by EITHER endpoint (unordered set).
    let mut kept_pairs: HashMap<(usize, usize), (f32, String)> = HashMap::new();
    for (_i, lst) in incident.iter_mut() {
        lst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    }
    // Iterate deterministically over nodes for stable tie behavior.
    let mut nodes: Vec<usize> = incident.keys().copied().collect();
    nodes.sort_unstable();
    for i in nodes {
        let lst = &incident[&i];
        for (w, j, src) in lst.iter().take(MERGE_TOPK) {
            let key = if i < *j { (i, *j) } else { (*j, i) };
            kept_pairs.insert(key, (*w, src.clone()));
        }
    }

    // Step 2: symmetric adjacency (reciprocity is automatic).
    let mut adj: HashMap<usize, HashMap<usize, (f32, String)>> = HashMap::new();
    for (&(i, j), (w, src)) in &kept_pairs {
        adj.entry(i).or_default().insert(j, (*w, src.clone()));
        adj.entry(j).or_default().insert(i, (*w, src.clone()));
    }

    // Step 3: HARD_CAP. Iteratively drop the globally weakest incident edge
    // of any over-degree node until all degrees <= HARD_CAP. Removing from
    // BOTH endpoints preserves symmetry.
    loop {
        let mut over: Vec<usize> = adj
            .iter()
            .filter(|(_, v)| v.len() > HARD_CAP)
            .map(|(k, _)| *k)
            .collect();
        if over.is_empty() {
            break;
        }
        over.sort_unstable();
        let mut weakest: Option<(f32, usize, usize)> = None;
        for &x in &over {
            if let Some(nbrs) = adj.get(&x) {
                for (&y, (w, _s)) in nbrs {
                    match weakest {
                        None => weakest = Some((*w, x, y)),
                        Some((bw, _, _)) if *w < bw => weakest = Some((*w, x, y)),
                        _ => {}
                    }
                }
            }
        }
        let Some((_, i, j)) = weakest else { break };
        if let Some(n) = adj.get_mut(&i) {
            n.remove(&j);
        }
        if let Some(n) = adj.get_mut(&j) {
            n.remove(&i);
        }
    }

    // Order each node's list by descending weight.
    let mut final_g: MergedGraph = HashMap::new();
    for (i, nbrs) in &adj {
        if nbrs.is_empty() {
            continue;
        }
        let mut ordered: Vec<(usize, f32, String)> =
            nbrs.iter().map(|(&j, (w, s))| (j, *w, s.clone())).collect();
        ordered.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));
        final_g.insert(*i, ordered);
    }

    Ok((final_g, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_dump() -> PathBuf {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("edges.json");
        // Leak the tempdir so the path stays valid for the test body.
        std::mem::forget(d);
        p
    }

    /// Build unit vectors so that `dot(v[i], v[j])` equals a chosen cosine.
    /// We use 2-D vectors on the unit circle: angle chosen s.t. cos = target.
    fn vec_for(angle: f32) -> Vec<f32> {
        vec![angle.cos(), angle.sin()]
    }

    fn a_map(pairs: &[(usize, usize, f64)]) -> HashMap<usize, HashMap<usize, f64>> {
        let mut m: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for &(i, j, c) in pairs {
            m.entry(i).or_default().insert(j, c);
            m.entry(j).or_default().insert(i, c);
        }
        m
    }

    fn b_map(pairs: &[(usize, usize, f32)]) -> HashMap<usize, HashMap<usize, f32>> {
        let mut m: HashMap<usize, HashMap<usize, f32>> = HashMap::new();
        for &(i, j, c) in pairs {
            m.entry(i).or_default().insert(j, c);
            m.entry(j).or_default().insert(i, c);
        }
        m
    }

    #[test]
    fn decision_table_ab_b_aveto_vetoa_belowb() {
        // Place 6 nodes; control pairwise cosines via angles from node 0.
        // node0 at angle 0. Others at angles whose cos vs node0 = target.
        // We only assert pairs (0,k).
        let papers: Vec<Paper> = (0..6)
            .map(|id| Paper {
                id: id as i64,
                file_path: format!("/x/{id}.md"),
                relpath: format!("00-General/{id}.md"),
                folder: "00-General".into(),
                basename: format!("{id}.md"),
                title: String::new(),
                tags: Default::default(),
                authors: Default::default(),
            })
            .collect();

        // cos(0,k) = cos(angle_k). Pick angles:
        //   k1: cos 0.70  -> >=0.55
        //   k2: cos 0.60  -> >=0.55
        //   k3: cos 0.45  -> [a_veto=0.40, <0.55]
        //   k4: cos 0.20  -> < a_veto
        //   k5: cos 0.20  -> < 0.55 (B candidate only)
        let m = vec![
            vec_for(0.0),
            vec_for(0.70_f32.acos()),
            vec_for(0.60_f32.acos()),
            vec_for(0.45_f32.acos()),
            vec_for(0.20_f32.acos()),
            vec_for(0.20_f32.acos()),
        ];
        let have = vec![true; 6];

        // A-pairs: (0,1) AB, (0,3) A-veto-pass, (0,4) veto-A.
        let a = a_map(&[(0, 1, 0.9), (0, 3, 0.9), (0, 4, 0.9)]);
        // B-pairs: (0,2) B, (0,5) below-B.
        let b = b_map(&[(0, 2, 0.60), (0, 5, 0.20)]);

        let dump = tmp_dump();
        let (g, st) = merge_graph(&papers, &a, &b, &m, &have, 0.40, &dump).unwrap();

        // src checks via final graph neighbor list of node 0.
        let n0: HashMap<usize, (f32, String)> = g[&0]
            .iter()
            .map(|(j, w, s)| (*j, (*w, s.clone())))
            .collect();
        assert_eq!(n0[&1].1, "AB");
        assert_eq!(n0[&2].1, "B");
        assert_eq!(n0[&3].1, "A");
        assert!(!n0.contains_key(&4), "veto-A dropped");
        assert!(!n0.contains_key(&5), "below-B dropped");

        // Uniform embedding-cosine weighting (NOT the a_cos 0.9).
        assert!((n0[&1].0 - 0.70).abs() < 1e-4);
        assert!((n0[&2].0 - 0.60).abs() < 1e-4);
        assert!((n0[&3].0 - 0.45).abs() < 1e-4);

        // Decision tallies.
        assert_eq!(st.keep_b_ab, 2); // (0,1)=AB, (0,2)=B
        assert_eq!(st.keep_a_veto_pass, 1); // (0,3)=A
        assert_eq!(st.veto_a, 1); // (0,4)
        assert_eq!(st.below_b, 1); // (0,5)
    }

    #[test]
    fn reciprocity_is_automatic_and_symmetric() {
        let papers: Vec<Paper> = (0..3)
            .map(|id| Paper {
                id: id as i64,
                file_path: format!("/x/{id}.md"),
                relpath: format!("00-General/{id}.md"),
                folder: "00-General".into(),
                basename: format!("{id}.md"),
                title: String::new(),
                tags: Default::default(),
                authors: Default::default(),
            })
            .collect();
        let m = vec![
            vec_for(0.0),
            vec_for(0.80_f32.acos()),
            vec_for(0.70_f32.acos()),
        ];
        let have = vec![true; 3];
        let a = a_map(&[]);
        let b = b_map(&[(0, 1, 0.80), (1, 2, 0.70)]);
        let dump = tmp_dump();
        let (g, _) = merge_graph(&papers, &a, &b, &m, &have, 0.40, &dump).unwrap();
        // Every edge present on both endpoints.
        for (&i, lst) in &g {
            for (j, _, _) in lst {
                assert!(
                    g[j].iter().any(|(k, _, _)| *k == i),
                    "edge {i}->{j} not reciprocated"
                );
            }
        }
    }

    #[test]
    fn drop_no_vec_when_endpoint_missing() {
        let papers: Vec<Paper> = (0..2)
            .map(|id| Paper {
                id: id as i64,
                file_path: format!("/x/{id}.md"),
                relpath: format!("00-General/{id}.md"),
                folder: "00-General".into(),
                basename: format!("{id}.md"),
                title: String::new(),
                tags: Default::default(),
                authors: Default::default(),
            })
            .collect();
        let m = vec![vec_for(0.0), vec_for(0.0)];
        let have = vec![true, false]; // node 1 has no vector.
        let a = a_map(&[(0, 1, 0.9)]);
        let b = b_map(&[]);
        let dump = tmp_dump();
        let (g, st) = merge_graph(&papers, &a, &b, &m, &have, 0.40, &dump).unwrap();
        assert!(g.is_empty());
        assert_eq!(st.keep_b_ab, 0);
        assert_eq!(st.veto_a, 0);
    }

    #[test]
    fn hard_cap_enforced_and_symmetry_preserved() {
        // One hub (node 0) connected to 14 leaves, all strong B edges with
        // varying weights. After cap, degree(0) must be <= HARD_CAP and the
        // graph must remain symmetric.
        let count = 15;
        let papers: Vec<Paper> = (0..count)
            .map(|id| Paper {
                id: id as i64,
                file_path: format!("/x/{id}.md"),
                relpath: format!("00-General/{id}.md"),
                folder: "00-General".into(),
                basename: format!("{id}.md"),
                title: String::new(),
                tags: Default::default(),
                authors: Default::default(),
            })
            .collect();
        // node0 at angle 0; leaves k at angles giving distinct cosines.
        let mut m = vec![vec_for(0.0)];
        let mut b_pairs: Vec<(usize, usize, f32)> = Vec::new();
        for k in 1..count {
            let c = 0.60_f32 + (k as f32) * 0.01; // 0.61..0.74, all >=0.55
            m.push(vec_for(c.acos()));
            b_pairs.push((0, k, c));
        }
        let have = vec![true; count];
        let a = a_map(&[]);
        let b = b_map(&b_pairs);
        let dump = tmp_dump();
        let (g, _) = merge_graph(&papers, &a, &b, &m, &have, 0.40, &dump).unwrap();

        assert!(g[&0].len() <= HARD_CAP, "degree {} > cap", g[&0].len());
        // Symmetry preserved after cap.
        for (&i, lst) in &g {
            for (j, _, _) in lst {
                assert!(g[j].iter().any(|(k, _, _)| *k == i));
            }
        }
        // Kept edges should be the strongest ones (descending by weight).
        let ws: Vec<f32> = g[&0].iter().map(|(_, w, _)| *w).collect();
        for w in ws.windows(2) {
            assert!(w[0] >= w[1], "neighbor list not weight-descending");
        }
    }
}
