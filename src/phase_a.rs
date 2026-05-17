//! Phase A: IDF-weighted cosine over (namespaced tags + authors).
//!
//! All numeric work in `f64` to match the Python `numpy.float64` path.

use std::collections::HashMap;

use crate::corpus::Paper;

/// Tag namespaces to KEEP (prefix match). Note the data uses
/// `FoS/Secondary/` although the original spec wrote `FoS/Second/` — both are
/// accepted.
const KEEP_TAG_PREFIXES: [&str; 8] = [
    "Survey/",
    "Method/",
    "Concept/",
    "Dataset/",
    "FoS/Primary/",
    "FoS/Second/",
    "FoS/Secondary/",
    "Venue/",
];
/// Tags excluded by exact match.
const EXCLUDE_EXACT: [&str; 1] = ["学術論文"];
/// Tag namespaces excluded by prefix.
const EXCLUDE_PREFIXES: [&str; 4] = ["Year/", "Lang/", "Cite/", "Type/"];

/// Minimum cosine for an A candidate.
pub const A_COSINE_MIN: f64 = 0.30;
/// A pair must share at least one term with df <= this to be a candidate.
pub const A_SPECIFIC_DF_MAX: usize = 200;
/// Per-node top-K A candidates kept.
pub const A_TOPK: usize = 8;

/// Decide whether a tag contributes a Phase-A term.
fn keep_tag(tag: &str) -> bool {
    if EXCLUDE_EXACT.contains(&tag) {
        return false;
    }
    if EXCLUDE_PREFIXES.iter().any(|p| tag.starts_with(p)) {
        return false;
    }
    KEEP_TAG_PREFIXES.iter().any(|p| tag.starts_with(p))
}

/// Phase-A result: directed `a_edges[i] -> {j: cosine}` plus stats.
pub struct PhaseAResult {
    /// `a_edges[i]` = list of `(j, cosine)` kept for node `i` (top-K by cos).
    pub a_edges: HashMap<usize, HashMap<usize, f64>>,
    /// Number of papers with >=1 A-edge.
    pub papers_with_edge: usize,
    /// Total undirected A-edges (deduplicated symmetric).
    pub total_edges: usize,
    /// Undirected edge weights (cosines) for percentile stats.
    pub weights: Vec<f64>,
    /// Distinct term count (size of df map).
    pub n_terms: usize,
    /// Number of terms with nonzero idf.
    pub n_terms_nonzero_idf: usize,
}

/// Compute Phase A. Mirrors the Python `phase_a` exactly:
/// inverted-index dot products over shared nonzero terms, specific-term gate
/// (df <= 200), cosine >= 0.30, per-node top-8 added symmetrically to both
/// endpoints' candidate lists.
pub fn phase_a(papers: &[Paper]) -> PhaseAResult {
    let n = papers.len();

    // term set per paper + document frequency.
    let mut terms_per_paper: Vec<Vec<String>> = Vec::with_capacity(n);
    let mut df: HashMap<String, usize> = HashMap::new();
    for p in papers {
        let mut terms: Vec<String> = Vec::new();
        for t in &p.tags {
            if keep_tag(t) {
                terms.push(t.clone());
            }
        }
        for a in &p.authors {
            terms.push(format!("author::{a}"));
        }
        // De-duplicate (Python used a set).
        terms.sort();
        terms.dedup();
        for t in &terms {
            *df.entry(t.clone()).or_insert(0) += 1;
        }
        terms_per_paper.push(terms);
    }

    // idf = max(0, ln(n / df)); df == n -> 0 -> term dropped.
    let mut idf: HashMap<String, f64> = HashMap::with_capacity(df.len());
    let mut n_terms_nonzero_idf = 0usize;
    for (t, &d) in &df {
        let v = (n as f64 / d as f64).ln().max(0.0);
        if v > 0.0 {
            n_terms_nonzero_idf += 1;
        }
        idf.insert(t.clone(), v);
    }

    // Per-paper weight vectors (drop idf == 0), L2 norms, inverted index.
    let mut norms = vec![0.0_f64; n];
    let mut inverted: HashMap<&str, Vec<(usize, f64)>> = HashMap::new();
    for (i, terms) in terms_per_paper.iter().enumerate() {
        let mut sumsq = 0.0_f64;
        for t in terms {
            let w = idf.get(t).copied().unwrap_or(0.0);
            if w > 0.0 {
                sumsq += w * w;
                inverted.entry(t.as_str()).or_default().push((i, w));
            }
        }
        norms[i] = sumsq.sqrt();
    }

    // Accumulate dot products only for pairs sharing >=1 nonzero term.
    let mut dots: HashMap<(usize, usize), f64> = HashMap::new();
    let mut shared_specific: HashMap<(usize, usize), bool> = HashMap::new();
    for (t, postings) in &inverted {
        if postings.len() < 2 {
            continue;
        }
        let specific = df.get(*t).copied().unwrap_or(0) <= A_SPECIFIC_DF_MAX;
        // Pairwise posting-list join (every unordered pair sharing this term).
        for (a, &(ia, wa)) in postings.iter().enumerate() {
            for &(ib, wb) in &postings[a + 1..] {
                let key = if ia < ib { (ia, ib) } else { (ib, ia) };
                *dots.entry(key).or_insert(0.0) += wa * wb;
                if specific {
                    shared_specific.insert(key, true);
                }
            }
        }
    }

    // Candidate generation: cos >= 0.30 AND shares a specific term.
    let mut candidates: HashMap<usize, Vec<(f64, usize)>> = HashMap::new();
    for (&(i, j), &dot) in &dots {
        if norms[i] == 0.0 || norms[j] == 0.0 {
            continue;
        }
        let cos = dot / (norms[i] * norms[j]);
        if cos >= A_COSINE_MIN && shared_specific.get(&(i, j)).copied().unwrap_or(false) {
            candidates.entry(i).or_default().push((cos, j));
            candidates.entry(j).or_default().push((cos, i));
        }
    }

    // Per-node top-K by cosine.
    let mut a_edges: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
    for (i, mut lst) in candidates {
        // Sort by cosine descending. Stable sort matches Python's list.sort.
        lst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
        let entry = a_edges.entry(i).or_default();
        for &(cos, j) in lst.iter().take(A_TOPK) {
            entry.insert(j, cos);
        }
    }

    // Symmetrise candidate relation for stats.
    let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let mut total_edges = 0usize;
    let mut weights: Vec<f64> = Vec::new();
    for (&i, nbrs) in &a_edges {
        for (&j, &cos) in nbrs {
            let key = if i < j { (i, j) } else { (j, i) };
            if seen.insert(key) {
                total_edges += 1;
                weights.push(cos);
            }
        }
    }

    let papers_with_edge = a_edges.values().filter(|m| !m.is_empty()).count();

    PhaseAResult {
        a_edges,
        papers_with_edge,
        total_edges,
        weights,
        n_terms: df.len(),
        n_terms_nonzero_idf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn paper(id: i64, tags: &[&str], authors: &[&str]) -> Paper {
        Paper {
            id,
            file_path: format!("/x/{id}.md"),
            relpath: format!("00-General/{id}.md"),
            folder: "00-General".into(),
            basename: format!("{id}.md"),
            title: String::new(),
            tags: tags.iter().map(|s| s.to_string()).collect::<HashSet<_>>(),
            authors: authors
                .iter()
                .map(|s| s.to_string())
                .collect::<HashSet<_>>(),
        }
    }

    #[test]
    fn term_in_all_docs_has_zero_idf_and_is_dropped() {
        // "Concept/x" appears in every doc -> idf 0 -> contributes nothing.
        // Only the unique shared specific tag should create an edge.
        let papers = vec![
            paper(0, &["Concept/x", "Method/shared"], &[]),
            paper(1, &["Concept/x", "Method/shared"], &[]),
            paper(2, &["Concept/x"], &[]),
        ];
        let r = phase_a(&papers);
        // Concept/x df==3==n -> idf 0. Method/shared df==2 -> idf>0, specific.
        // Papers 0 and 1 share Method/shared only -> cosine 1.0 edge.
        assert_eq!(r.a_edges[&0].get(&1).copied(), Some(1.0));
        assert_eq!(r.a_edges[&1].get(&0).copied(), Some(1.0));
        // Paper 2 has only the zero-idf term -> no edges.
        assert!(r.a_edges.get(&2).map(|m| m.is_empty()).unwrap_or(true));
    }

    #[test]
    fn coarse_only_shared_term_forms_no_edge() {
        // A term shared by 201+ docs (df > 200) is not "specific": even if its
        // cosine clears 0.30, no candidate is formed without a specific term.
        let mut papers: Vec<Paper> = Vec::new();
        for i in 0..205 {
            papers.push(paper(i, &["Concept/coarse"], &[]));
        }
        let r = phase_a(&papers);
        // df(Concept/coarse) = 205 > 200 -> not specific -> no edges at all.
        assert!(r.total_edges == 0);
        assert!(r.a_edges.values().all(|m| m.is_empty()));
    }

    #[test]
    fn rare_specific_shared_term_forms_edge() {
        // Papers 0 and 1 share two rare specific terms; their per-paper
        // vectors are otherwise identical, so cosine == 1.0 >= 0.30 and the
        // specific-term gate (df <= 200) passes -> an edge is formed.
        let papers = vec![
            paper(0, &["Method/rare", "Concept/shared"], &[]),
            paper(1, &["Method/rare", "Concept/shared"], &[]),
            paper(2, &["Venue/C"], &[]),
            paper(3, &["Venue/D"], &[]),
        ];
        let r = phase_a(&papers);
        // Method/rare & Concept/shared df==2 (specific) -> edge between 0,1.
        assert!(r.a_edges.get(&0).and_then(|m| m.get(&1)).is_some());
        assert!(r.a_edges.get(&1).and_then(|m| m.get(&0)).is_some());
        // Papers 2 and 3 share nothing -> no edge.
        assert!(r.a_edges.get(&2).map(|m| m.is_empty()).unwrap_or(true));
    }

    #[test]
    fn shared_specific_term_below_cosine_min_forms_no_edge() {
        // Papers 0,1 share one rare specific term but each also has a unique
        // high-idf tag, so their cosine (~0.20) is below A_COSINE_MIN=0.30.
        // The specific-term gate passes, yet no candidate is kept.
        let papers = vec![
            paper(0, &["Method/rare", "Venue/A"], &[]),
            paper(1, &["Method/rare", "Venue/B"], &[]),
            paper(2, &["Venue/C"], &[]),
            paper(3, &["Venue/D"], &[]),
        ];
        let r = phase_a(&papers);
        assert!(r.a_edges.get(&0).map(|m| m.is_empty()).unwrap_or(true));
        assert_eq!(r.total_edges, 0);
    }

    #[test]
    fn excluded_tags_do_not_contribute() {
        let papers = vec![
            paper(0, &["学術論文", "Year/2020", "Type/journal"], &[]),
            paper(1, &["学術論文", "Year/2020", "Type/journal"], &[]),
        ];
        let r = phase_a(&papers);
        // All shared tags are excluded -> no terms -> no edges.
        assert_eq!(r.total_edges, 0);
        assert_eq!(r.n_terms, 0);
    }

    #[test]
    fn author_term_contributes() {
        let papers = vec![
            paper(0, &[], &["Alice"]),
            paper(1, &[], &["Alice"]),
            paper(2, &[], &["Bob"]),
        ];
        let r = phase_a(&papers);
        // author::Alice df==2 (specific) -> edge between 0 and 1.
        assert!(r.a_edges.get(&0).and_then(|m| m.get(&1)).is_some());
    }
}
