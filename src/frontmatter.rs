//! Idempotent `related-documents:` YAML-frontmatter reconstruction.
//!
//! Every run fully rebuilds the block from scratch, so re-running on an
//! already-written file produces byte-identical output (idempotent).

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;

/// Build the replacement block lines for the given targets.
/// No targets → a single inline `related-documents: []` line.
fn build_block(targets: &[String]) -> Vec<String> {
    if targets.is_empty() {
        return ["related-documents: []".to_string()].to_vec();
    }
    let mut out = Vec::with_capacity(targets.len() + 1);
    out.push("related-documents:".to_string());
    for t in targets {
        out.push(format!("  - \"[[{t}]]\""));
    }
    out
}

/// Reconstruct the frontmatter of `original`, replacing the
/// `related-documents:` block with `new_targets`.
///
/// Returns `(new_text, changed)`. Mirrors the Python `reconstruct_file`:
/// - line0 trimmed must be `---`, else returned unchanged.
/// - find next line trimmed `---` (frontmatter end), else unchanged.
/// - key regex `^related-documents:\s*(\[\s*\])?\s*$`, item regex `^\s+-\s`.
/// - on key match: skip key line + consecutive item lines, emit rebuilt block.
/// - if key never seen: append rebuilt block before the closing `---`.
/// - reassemble `["---"] + new_fm + ["---"] + rest`, join with `\n`.
pub fn reconstruct(original: &str, new_targets: &[String]) -> (String, bool) {
    let lines: Vec<&str> = original.split('\n').collect();
    if lines.first().map(|l| l.trim()) != Some("---") {
        return (original.to_string(), false);
    }
    let mut fm_end: Option<usize> = None;
    for (idx, ln) in lines.iter().enumerate().skip(1) {
        if ln.trim() == "---" {
            fm_end = Some(idx);
            break;
        }
    }
    let Some(fm_end) = fm_end else {
        return (original.to_string(), false);
    };

    let fm = &lines[1..fm_end];
    let key_re = Regex::new(r"^related-documents:\s*(\[\s*\])?\s*$").unwrap();
    let item_re = Regex::new(r"^\s+-\s").unwrap();

    let mut new_fm: Vec<String> = Vec::with_capacity(fm.len() + new_targets.len() + 1);
    let mut i = 0usize;
    let mut inserted = false;
    while i < fm.len() {
        let ln = fm[i];
        if key_re.is_match(ln) {
            // Skip key line + following list-item lines (block form).
            i += 1;
            while i < fm.len() && item_re.is_match(fm[i]) {
                i += 1;
            }
            new_fm.extend(build_block(new_targets));
            inserted = true;
            continue;
        }
        new_fm.push(ln.to_string());
        i += 1;
    }

    if !inserted {
        new_fm.extend(build_block(new_targets));
    }

    let mut rebuilt: Vec<String> = Vec::with_capacity(new_fm.len() + lines.len());
    rebuilt.push("---".to_string());
    rebuilt.extend(new_fm);
    rebuilt.push("---".to_string());
    for ln in &lines[fm_end + 1..] {
        rebuilt.push((*ln).to_string());
    }
    let new_text = rebuilt.join("\n");
    let changed = new_text != original;
    (new_text, changed)
}

/// Read a file, reconstruct its block, and atomically write iff changed.
/// Returns whether the file was changed.
pub fn reconstruct_file(file_path: &Path, new_targets: &[String]) -> Result<bool> {
    let original =
        fs::read_to_string(file_path).with_context(|| format!("read {}", file_path.display()))?;
    let (new_text, changed) = reconstruct(&original, new_targets);
    if changed {
        atomic_write(file_path, &new_text)?;
    }
    Ok(changed)
}

/// Atomic write: temp file `<path>.tmp.relate` then rename over the target.
fn atomic_write(file_path: &Path, text: &str) -> Result<()> {
    let tmp = file_path.with_extension(
        // Append ".tmp.relate" while preserving the original full name.
        match file_path.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("{ext}.tmp.relate"),
            None => "tmp.relate".to_string(),
        },
    );
    fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
    fs::rename(&tmp, file_path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), file_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_inline_empty_form() {
        let original = "\
---
title: x
related-documents: []
tags: [a]
---
body line 1
body line 2";
        let (out, changed) = reconstruct(original, &["Foo".into(), "Bar".into()]);
        assert!(changed);
        assert!(out.contains("related-documents:\n  - \"[[Foo]]\"\n  - \"[[Bar]]\""));
        // Other keys preserved verbatim.
        assert!(out.contains("title: x"));
        assert!(out.contains("tags: [a]"));
        // Body preserved exactly.
        assert!(out.ends_with("body line 1\nbody line 2"));
    }

    #[test]
    fn replaces_block_form() {
        let original = "\
---
title: x
related-documents:
  - \"[[Old1]]\"
  - \"[[Old2]]\"
status: done
---
content";
        let (out, changed) = reconstruct(original, &["New".into()]);
        assert!(changed);
        assert!(out.contains("related-documents:\n  - \"[[New]]\""));
        assert!(!out.contains("Old1"));
        assert!(!out.contains("Old2"));
        assert!(out.contains("status: done"));
        assert!(out.ends_with("---\ncontent"));
    }

    #[test]
    fn appends_when_key_absent() {
        let original = "\
---
title: x
tags: [a]
---
body";
        let (out, changed) = reconstruct(original, &["T".into()]);
        assert!(changed);
        // Block appended at the END of frontmatter, before closing ---.
        let expected = "\
---
title: x
tags: [a]
related-documents:
  - \"[[T]]\"
---
body";
        assert_eq!(out, expected);
    }

    #[test]
    fn empty_targets_writes_inline_empty() {
        let original = "\
---
related-documents:
  - \"[[Gone]]\"
---
x";
        let (out, _) = reconstruct(original, &[]);
        assert!(out.contains("related-documents: []"));
        assert!(!out.contains("Gone"));
    }

    #[test]
    fn non_frontmatter_file_unchanged() {
        let original = "no frontmatter here\njust text\n";
        let (out, changed) = reconstruct(original, &["X".into()]);
        assert!(!changed);
        assert_eq!(out, original);
    }

    #[test]
    fn unterminated_frontmatter_unchanged() {
        let original = "---\ntitle: x\nno closing fence\n";
        let (out, changed) = reconstruct(original, &["X".into()]);
        assert!(!changed);
        assert_eq!(out, original);
    }

    #[test]
    fn idempotent_reconstruct_twice_is_identical() {
        let original = "\
---
title: x
related-documents: []
extra: y
---
body content
more body";
        let targets = vec!["A".to_string(), "B".to_string()];
        let (first, c1) = reconstruct(original, &targets);
        assert!(c1);
        let (second, c2) = reconstruct(&first, &targets);
        // Re-running on already-written output makes NO change.
        assert!(!c2, "second reconstruct must be a no-op");
        assert_eq!(first, second);
    }

    #[test]
    fn idempotent_empty_targets() {
        let original = "---\nk: v\n---\nbody";
        let (first, _) = reconstruct(original, &[]);
        let (second, c2) = reconstruct(&first, &[]);
        assert!(!c2);
        assert_eq!(first, second);
    }

    #[test]
    fn all_other_bytes_preserved() {
        let original = "\
---
title: \"複雑な タイトル\"
authors:
  - Alice
  - Bob
related-documents:
  - \"[[Old]]\"
nested:
  key: value
---
# 本文

段落です。

```rust
fn main() {}
```
末尾";
        let (out, _) = reconstruct(original, &["New".into()]);
        // Everything except the related-documents block must be byte-identical.
        assert!(out.contains("title: \"複雑な タイトル\""));
        assert!(out.contains("  - Alice\n  - Bob"));
        assert!(out.contains("nested:\n  key: value"));
        assert!(out.contains("```rust\nfn main() {}\n```"));
        assert!(out.ends_with("末尾"));
        assert!(out.contains("related-documents:\n  - \"[[New]]\""));
    }
}
