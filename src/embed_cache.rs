//! Content-hash embedding cache.
//!
//! Key = lowercase hex SHA-256 of the **exact** embed input string (the empty
//! string is hashed as a single space `" "`, matching the Python `inputs[i] or
//! " "`). Identical embed text -> reuse vector, so re-runs are free and the
//! analysis pass and a later `--apply` see the SAME vectors (no drift).
//!
//! This is a **separate** file from the Python `.npz` cache (different format);
//! the Python cache is never read. The first Rust run therefore re-embeds the
//! full corpus.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// sha256-hex -> L2-normalized f32 vector.
pub type EmbedCache = HashMap<String, Vec<f32>>;

/// Hash an embed input. Empty string is hashed as `" "` (Python parity).
pub fn hash_input(s: &str) -> String {
    let mut hasher = Sha256::new();
    if s.is_empty() {
        hasher.update(b" ");
    } else {
        hasher.update(s.as_bytes());
    }
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// Load the cache. A missing or unreadable/corrupt cache yields an empty map
/// (matching the Python's defensive `except: return {}`).
pub fn load(path: &Path) -> EmbedCache {
    let Ok(bytes) = fs::read(path) else {
        return EmbedCache::new();
    };
    match bincode::deserialize::<Vec<(String, Vec<f32>)>>(&bytes) {
        Ok(v) => v.into_iter().collect(),
        Err(_) => EmbedCache::new(),
    }
}

/// Persist the cache atomically (write temp file, then rename).
pub fn save(path: &Path, cache: &EmbedCache) -> Result<()> {
    if cache.is_empty() {
        return Ok(());
    }
    let v: Vec<(String, Vec<f32>)> = cache
        .iter()
        .map(|(k, val)| (k.clone(), val.clone()))
        .collect();
    let bytes = bincode::serialize(&v).context("serialize embed cache")?;
    let tmp: PathBuf = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_hashes_as_space() {
        // Python: hashlib.sha256((inputs[i] or " ").encode()).hexdigest()
        assert_eq!(hash_input(""), hash_input(" "));
        assert_eq!(hash_input("").len(), 64);
        assert!(hash_input("").chars().all(|c| c.is_ascii_hexdigit()));
        // Known SHA-256 of a single ASCII space.
        assert_eq!(
            hash_input(""),
            "36a9e7f1c95b82ffb99743e0c5c4ce95d83c9a430aac59f84ef3cbfab6145068"
        );
    }

    #[test]
    fn hash_is_stable_and_lowercase() {
        let h = hash_input("hello");
        assert_eq!(h, hash_input("hello"));
        assert_eq!(h, h.to_lowercase());
        // Known SHA-256 of "hello".
        assert_eq!(
            h,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn roundtrip_save_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.bin");
        let mut c = EmbedCache::new();
        c.insert("abc".into(), vec![0.1, 0.2, 0.3]);
        save(&path, &c).unwrap();
        let loaded = load(&path);
        assert_eq!(loaded.get("abc"), Some(&vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn missing_or_corrupt_yields_empty() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.bin");
        assert!(load(&missing).is_empty());
        let corrupt = dir.path().join("bad.bin");
        fs::write(&corrupt, b"not bincode").unwrap();
        assert!(load(&corrupt).is_empty());
    }

    #[test]
    fn empty_cache_not_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.bin");
        save(&path, &EmbedCache::new()).unwrap();
        assert!(!path.exists());
    }
}
