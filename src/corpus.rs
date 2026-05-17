//! Corpus loading from the obsidian-paper-cache SQLite DB and the
//! list-index-keyed link-target resolver.
//!
//! Identity rule (load-bearing): every downstream graph (Phase A, Phase B,
//! merge) is keyed by the paper's **position in the loaded `Vec<Paper>`**
//! (`0..n-1`), never by the DB `id`. The resolver below uses the same identity.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use rusqlite::Connection;

use crate::config::Config;

/// One paper-report note loaded from the cache DB (and confirmed on disk).
#[derive(Debug, Clone)]
pub struct Paper {
    /// DB id (used only to join `tags`/`authors`; NOT a graph key).
    pub id: i64,
    /// Absolute path to the markdown file.
    pub file_path: String,
    /// Path relative to the reports dir, e.g. `02-Security/Foo.md`.
    pub relpath: String,
    /// Folder component (`00-General` etc.).
    pub folder: String,
    /// Basename including the `.md` extension.
    pub basename: String,
    /// Note title (may be empty).
    pub title: String,
    /// Namespaced tags.
    pub tags: HashSet<String>,
    /// Author display names (trimmed, non-empty).
    pub authors: HashSet<String>,
}

/// Compute the relpath of `file_path` against the reports dir.
///
/// Mirrors Python's `os.path.relpath(fp, REPORTS_DIR)` for the only shape we
/// care about (a descendant path): strip the directory prefix and a separator.
/// Returns `None` if `file_path` is not under the reports dir.
fn relpath_under_reports(file_path: &str, reports_dir: &Path) -> Option<String> {
    let base_str = reports_dir.to_string_lossy();
    let p = file_path.strip_prefix(base_str.as_ref())?;
    let p = p.strip_prefix('/').unwrap_or(p);
    Some(p.to_string())
}

/// Load the corpus from the cache DB.
///
/// Behavior preserved exactly from the Python `load_corpus`:
/// - SQL `file_path LIKE '%<reports_rel>/<folder>/%'` for each folder.
/// - maxdepth-1 filter: relpath splits into exactly `<folder>/<base>`,
///   `<folder>` in `cfg.folders`, `base` not starting with `_`, ending `.md`.
/// - **the file must exist on disk** — stale DB rows for renamed/deleted notes
///   are excluded entirely (prevents a mid-apply crash and dead link targets).
/// - tags/authors joined by DB id afterwards.
pub fn load_corpus(conn: &Connection, cfg: &Config) -> Result<Vec<Paper>> {
    let reports_rel = &cfg.reports_rel;
    let like_clauses: Vec<String> = cfg
        .folders
        .iter()
        .map(|f| format!("file_path LIKE '%{reports_rel}/{f}/%'"))
        .collect();
    let sql = format!(
        "SELECT id, file_path, title FROM papers WHERE {}",
        like_clauses.join(" OR ")
    );

    let folder_set: HashSet<&str> = cfg.folders.iter().map(String::as_str).collect();
    let reports_dir = cfg.reports_dir();

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        let file_path: String = row.get(1)?;
        let title: Option<String> = row.get(2)?;
        Ok((id, file_path, title.unwrap_or_default()))
    })?;

    let mut papers: Vec<Paper> = Vec::new();
    for row in rows {
        let (id, file_path, title) = row?;
        let Some(rel) = relpath_under_reports(&file_path, &reports_dir) else {
            continue;
        };
        let parts: Vec<&str> = rel.split('/').collect();
        if parts.len() != 2 {
            continue;
        }
        let folder = parts[0];
        let base = parts[1];
        if !folder_set.contains(folder) {
            continue;
        }
        if base.starts_with('_') || !base.ends_with(".md") {
            continue;
        }
        if !Path::new(&file_path).exists() {
            // Stale cache row (note renamed/deleted). Exclude entirely so it
            // is neither a writer (no crash) nor a link target (no dead link).
            continue;
        }
        papers.push(Paper {
            id,
            file_path,
            relpath: rel.clone(),
            folder: folder.to_string(),
            basename: base.to_string(),
            title,
            tags: HashSet::new(),
            authors: HashSet::new(),
        });
    }

    // Join tags/authors by DB id.
    let mut by_id: HashMap<i64, usize> = HashMap::new();
    for (idx, p) in papers.iter().enumerate() {
        by_id.insert(p.id, idx);
    }

    {
        let mut stmt = conn.prepare("SELECT paper_id, tag FROM tags")?;
        let mut q = stmt.query([])?;
        while let Some(r) = q.next()? {
            let pid: i64 = r.get(0)?;
            let tag: String = r.get(1)?;
            if let Some(&idx) = by_id.get(&pid) {
                papers[idx].tags.insert(tag);
            }
        }
    }
    {
        let mut stmt = conn.prepare("SELECT paper_id, name FROM authors")?;
        let mut q = stmt.query([])?;
        while let Some(r) = q.next()? {
            let pid: i64 = r.get(0)?;
            let name: Option<String> = r.get(1)?;
            if let (Some(&idx), Some(name)) = (by_id.get(&pid), name) {
                let name = name.trim();
                if !name.is_empty() {
                    papers[idx].authors.insert(name.to_string());
                }
            }
        }
    }

    Ok(papers)
}

/// Build the list-index -> link-target map.
///
/// Keyed by list position (0..n-1) — the same identity used by the A/B edge
/// graphs, NOT by DB id. Target = basename without `.md` if that stem is unique
/// across the corpus, else `<reports_rel>/<folder>/<stem>`.
pub fn build_target_resolver(papers: &[Paper], cfg: &Config) -> HashMap<usize, String> {
    let reports_rel = &cfg.reports_rel;
    let mut base_count: HashMap<&str, usize> = HashMap::new();
    for p in papers {
        let stem = stem_of(&p.basename);
        *base_count.entry(stem).or_insert(0) += 1;
    }

    let mut out: HashMap<usize, String> = HashMap::with_capacity(papers.len());
    for (idx, p) in papers.iter().enumerate() {
        let stem = stem_of(&p.basename);
        let target = if base_count.get(stem).copied().unwrap_or(0) == 1 {
            stem.to_string()
        } else {
            format!("{}/{}/{}", reports_rel, p.folder, stem)
        };
        out.insert(idx, target);
    }
    out
}

/// Basename without the trailing `.md` (Python's `basename[:-3]`).
fn stem_of(basename: &str) -> &str {
    basename.strip_suffix(".md").unwrap_or(basename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Explicit, non-personal test config. Mirrors the prior default
    /// constants (`研究/98_論文レポート`, the three folders) but with a
    /// synthetic vault root so no real path is referenced.
    fn test_cfg() -> Config {
        Config::from_parts(
            Some("/test/vault".into()),
            None,
            None, // reports_rel default == prior REPORTS_REL constant.
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            None,
            Some("/synthetic-home".into()),
        )
        .unwrap()
    }

    fn mk(id: i64, folder: &str, base: &str) -> Paper {
        Paper {
            id,
            file_path: format!("/x/{folder}/{base}"),
            relpath: format!("{folder}/{base}"),
            folder: folder.to_string(),
            basename: base.to_string(),
            title: String::new(),
            tags: HashSet::new(),
            authors: HashSet::new(),
        }
    }

    #[test]
    fn unique_stem_resolves_to_basename() {
        let cfg = test_cfg();
        let papers = vec![
            mk(1, "00-General", "Alpha.md"),
            mk(2, "01-Simulation", "Beta.md"),
        ];
        let r = build_target_resolver(&papers, &cfg);
        assert_eq!(r[&0], "Alpha");
        assert_eq!(r[&1], "Beta");
    }

    #[test]
    fn colliding_stem_resolves_to_vault_relative() {
        let cfg = test_cfg();
        let rel = &cfg.reports_rel;
        let papers = vec![
            mk(1, "00-General", "Dup.md"),
            mk(2, "02-Security", "Dup.md"),
            mk(3, "01-Simulation", "Unique.md"),
        ];
        let r = build_target_resolver(&papers, &cfg);
        assert_eq!(r[&0], format!("{rel}/00-General/Dup"));
        assert_eq!(r[&1], format!("{rel}/02-Security/Dup"));
        assert_eq!(r[&2], "Unique");
    }

    #[test]
    fn relpath_under_reports_strips_prefix() {
        let cfg = test_cfg();
        let reports_dir = cfg.reports_dir();
        let fp = format!("{}/02-Security/Foo.md", reports_dir.display());
        assert_eq!(
            relpath_under_reports(&fp, &reports_dir).as_deref(),
            Some("02-Security/Foo.md")
        );
        assert_eq!(
            relpath_under_reports("/somewhere/else.md", &reports_dir),
            None
        );
    }
}
