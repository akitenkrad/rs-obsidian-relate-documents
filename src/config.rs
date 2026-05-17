//! Runtime configuration, externalized from source.
//!
//! Every machine/vault-specific value (vault root, reports path, included
//! folders, protected relpaths, cache directory, cache-DB path) is resolved
//! here from CLI flags, environment variables, and non-identifying defaults.
//!
//! The vault root is **required** and has **no default** — there is no
//! personal path anywhere in the repository. Resolution order for it is:
//! CLI flag → env `OBSIDIAN_VAULT_ROOT` → hard error.
//!
//! Resolution logic lives in [`Config::from_parts`] so it is unit-testable
//! without touching real process/env state; the CLI layer is a thin caller.

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};

/// Environment variable consulted when `--vault-root` is not given.
pub const ENV_VAULT_ROOT: &str = "OBSIDIAN_VAULT_ROOT";

/// Default reports directory, relative to the vault root.
///
/// Non-identifying generic default for this tool's purpose (a folder name),
/// so it is fine to keep as a constant default.
pub const DEFAULT_REPORTS_REL: &str = "研究/98_論文レポート";

/// Default report sub-folders included in the corpus.
pub const DEFAULT_FOLDERS: [&str; 3] = ["00-General", "01-Simulation", "02-Security"];

/// Embedding-cache filename (joined onto the cache directory).
pub const EMBED_CACHE_FILE: &str = "relate_emb_cache_rs.bin";

/// Edge-decision-dump filename (joined onto the cache directory).
pub const EDGE_DUMP_FILE: &str = "relate_edges_rs.json";

/// Resolved runtime configuration, threaded by reference through the pipeline.
#[derive(Debug, Clone)]
pub struct Config {
    /// Vault root on disk (required; no default).
    pub vault_root: PathBuf,
    /// Reports directory, relative to [`Config::vault_root`].
    pub reports_rel: String,
    /// Report sub-folders included in the corpus.
    pub folders: Vec<String>,
    /// Protected relpaths (relative to `reports_rel`): never written, but may
    /// still be link targets from other notes. Default = empty.
    pub protected_relpaths: Vec<String>,
    /// Directory holding the embedding cache + edge dump.
    pub cache_dir: PathBuf,
    /// Cache DB path (`obsidian-paper-cache` SQLite DB).
    pub db_path: PathBuf,
}

impl Config {
    /// Resolve a [`Config`] from already-extracted parts.
    ///
    /// This is the single place resolution rules live, so it can be unit
    /// tested without real env/process state:
    ///
    /// - `vault_root`: `cli_vault_root` → `env_vault_root` → hard error.
    /// - `reports_rel`: `cli_reports_rel` → [`DEFAULT_REPORTS_REL`].
    /// - `folders`: `cli_folders` if non-empty → [`DEFAULT_FOLDERS`].
    /// - `protected`: `cli_protected` (default empty).
    /// - `cache_dir`: `cli_cache_dir` → [`std::env::temp_dir`]. Created if missing.
    /// - `db_path`: `cli_db_path` → `<home>/.cache/obsidian-paper-cache/papers.db`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        cli_vault_root: Option<String>,
        env_vault_root: Option<String>,
        cli_reports_rel: Option<String>,
        cli_folders: Vec<String>,
        cli_protected: Vec<String>,
        cli_cache_dir: Option<String>,
        default_cache_dir: PathBuf,
        cli_db_path: Option<String>,
        home_dir: Option<String>,
    ) -> Result<Self> {
        let vault_root = cli_vault_root.or(env_vault_root).ok_or_else(|| {
            anyhow!("vault root required: pass --vault-root or set {ENV_VAULT_ROOT}")
        })?;

        let reports_rel = cli_reports_rel.unwrap_or_else(|| DEFAULT_REPORTS_REL.to_string());

        let folders = if cli_folders.is_empty() {
            DEFAULT_FOLDERS.iter().map(|s| (*s).to_string()).collect()
        } else {
            cli_folders
        };

        let cache_dir = cli_cache_dir
            .map(PathBuf::from)
            .unwrap_or(default_cache_dir);

        let db_path = match cli_db_path {
            Some(p) => PathBuf::from(p),
            None => {
                let home = home_dir.context("HOME not set")?;
                PathBuf::from(home).join(".cache/obsidian-paper-cache/papers.db")
            }
        };

        Ok(Config {
            vault_root: PathBuf::from(vault_root),
            reports_rel,
            folders,
            protected_relpaths: cli_protected,
            cache_dir,
            db_path,
        })
    }

    /// Absolute path to the reports directory (`<vault_root>/<reports_rel>`).
    pub fn reports_dir(&self) -> PathBuf {
        self.vault_root.join(&self.reports_rel)
    }

    /// Absolute path to the embedding cache file.
    pub fn embed_cache_path(&self) -> PathBuf {
        self.cache_dir.join(EMBED_CACHE_FILE)
    }

    /// Absolute path to the edge-decision dump file.
    pub fn edge_dump_path(&self) -> PathBuf {
        self.cache_dir.join(EDGE_DUMP_FILE)
    }

    /// Create the cache directory if it does not yet exist.
    pub fn ensure_cache_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.cache_dir)
            .with_context(|| format!("create cache dir {}", self.cache_dir.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_vault_root_wins_over_env() {
        let c = Config::from_parts(
            Some("/cli/vault".into()),
            Some("/env/vault".into()),
            None,
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            None,
            Some("/synthetic-home".into()),
        )
        .unwrap();
        assert_eq!(c.vault_root, PathBuf::from("/cli/vault"));
    }

    #[test]
    fn env_vault_root_used_when_no_cli() {
        let c = Config::from_parts(
            None,
            Some("/env/vault".into()),
            None,
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            None,
            Some("/synthetic-home".into()),
        )
        .unwrap();
        assert_eq!(c.vault_root, PathBuf::from("/env/vault"));
    }

    #[test]
    fn missing_vault_root_is_hard_error() {
        let err = Config::from_parts(
            None,
            None,
            None,
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            None,
            Some("/synthetic-home".into()),
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("vault root required"), "got: {msg}");
        assert!(msg.contains(ENV_VAULT_ROOT), "got: {msg}");
    }

    #[test]
    fn defaults_reproduce_prior_constants() {
        let c = Config::from_parts(
            Some("/v".into()),
            None,
            None,
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            None,
            Some("/synthetic-home".into()),
        )
        .unwrap();
        // Prior REPORTS_REL constant.
        assert_eq!(c.reports_rel, "研究/98_論文レポート");
        // Prior FOLDERS constant.
        assert_eq!(
            c.folders,
            vec!["00-General", "01-Simulation", "02-Security"]
        );
        // Prior PROTECTED_RELPATHS default is now empty (no personal data).
        assert!(c.protected_relpaths.is_empty());
        // Prior db_path() behavior: $HOME-derived default.
        assert_eq!(
            c.db_path,
            PathBuf::from("/synthetic-home/.cache/obsidian-paper-cache/papers.db")
        );
        // Prior cache filenames preserved, joined onto the cache dir.
        assert_eq!(
            c.embed_cache_path(),
            PathBuf::from("/tmp/relate_emb_cache_rs.bin")
        );
        assert_eq!(
            c.edge_dump_path(),
            PathBuf::from("/tmp/relate_edges_rs.json")
        );
        // reports_dir joins vault_root + reports_rel like the old constant pair.
        assert_eq!(
            c.reports_dir(),
            PathBuf::from("/v").join("研究/98_論文レポート")
        );
    }

    #[test]
    fn explicit_overrides_take_effect() {
        let c = Config::from_parts(
            Some("/v".into()),
            None,
            Some("custom/reports".into()),
            vec!["A".into(), "B".into()],
            vec!["A/note.md".into()],
            Some("/my/cache".into()),
            PathBuf::from("/tmp"),
            Some("/db/papers.db".into()),
            Some("/synthetic-home".into()),
        )
        .unwrap();
        assert_eq!(c.reports_rel, "custom/reports");
        assert_eq!(c.folders, vec!["A", "B"]);
        assert_eq!(c.protected_relpaths, vec!["A/note.md"]);
        assert_eq!(c.cache_dir, PathBuf::from("/my/cache"));
        assert_eq!(c.db_path, PathBuf::from("/db/papers.db"));
        assert_eq!(
            c.embed_cache_path(),
            PathBuf::from("/my/cache/relate_emb_cache_rs.bin")
        );
    }

    #[test]
    fn cli_db_path_overrides_home_default() {
        let c = Config::from_parts(
            Some("/v".into()),
            None,
            None,
            vec![],
            vec![],
            None,
            PathBuf::from("/tmp"),
            Some("/explicit/db.sqlite".into()),
            None, // no HOME, but cli db path provided so it's fine.
        )
        .unwrap();
        assert_eq!(c.db_path, PathBuf::from("/explicit/db.sqlite"));
    }
}
