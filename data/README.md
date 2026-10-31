# Data storage policy

Nothing under `data/` except this file and the `.gitkeep` placeholders is ever committed.

| Directory | Contents | How it is produced |
|---|---|---|
| `data/raw/` | Public datasets exactly as downloaded (zip/CSV/JSON) | `network-anomaly datasets download <name>` (pinned files, checksums verified) |
| `data/processed/` | Public datasets converted to the toolkit's canonical schema | `network-anomaly datasets prepare <name>` |
| `data/synthetic/` | Synthetic datasets with ground-truth labels and metadata | `network-anomaly generate` (deterministic from the seed) |

## Rules

- **No real customer traffic, logs or secrets in the repository.** Infrastructure metrics and network-flow
  records can identify people and systems; keep them outside version control and restrict access to the
  machines and people who need them.
- **Public datasets are not redistributed.** Each user downloads them from the official source and accepts
  that source's licence and citation terms.
- **Synthetic data is reproducible**, so it is regenerated rather than stored: every file is accompanied by
  a `.metadata.json` recording the seed, configuration, generator version and SHA-256 checksum.
- **Minimise sensitive fields.** Drop or pseudonymise IP addresses and hostnames as early as possible;
  they are never used as model features.
- **Retention.** Delete raw extracts once the processed files you need exist, and apply your organisation's
  retention policy to anything derived from production systems.
