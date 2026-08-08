# Support Knowledge Corpus

Downloaded 2026-08-09. **3,396 markdown/MDX documents, ~19 MB of text** (78 MB on disk including
git metadata). All sources are shallow, blob-filtered, sparse clones — `.git` is retained on purpose
because `git log` is the source of the `last_updated` metadata field.

## Sources

| Path | Source | License | Docs |
|---|---|---|---|
| `raw/fastapi/docs/en/docs` | [fastapi/fastapi](https://github.com/fastapi/fastapi) | MIT | 155 |
| `raw/k8s-website/content/en/docs` | [kubernetes/website](https://github.com/kubernetes/website) | CC BY 4.0 | 521 |
| `raw/posthog/contents/handbook` | [PostHog/posthog.com](https://github.com/PostHog/posthog.com) | MIT (`/contents/` only) | 364 |
| `raw/posthog/contents/docs` | same | MIT (`/contents/` only) | 1,873 |
| `raw/posthog/contents/tutorials` | same | MIT (`/contents/` only) | 195 |
| `raw/zulip/starlight_help/src/content/docs` | [zulip/zulip](https://github.com/zulip/zulip) | Apache 2.0 | 259 |
| `raw/zulip/api_docs` | same | Apache 2.0 | 29 |

PostHog's root `LICENSE` restricts the website design but grants MIT terms for everything under
`/contents/`, which is all we checked out.

## Coverage against the required document types

| Required type | Where it comes from |
|---|---|
| Product FAQs | Zulip help center — ~260 "how do I X" support articles |
| Troubleshooting guides | `k8s-website/.../tasks/debug/`, PostHog docs troubleshooting pages |
| Onboarding docs | `posthog/contents/handbook/` (people, engineering, new-starter) |
| API documentation | FastAPI reference, Zulip `api_docs`, PostHog API docs |
| Release notes | `fastapi/docs/en/docs/release-notes.md`, PostHog changelog entries |
| Policy pages | `posthog/contents/handbook/` (compensation, leave, security, brand) |

## Why this mix (retrieval difficulty is deliberate)

- **Overlapping information** — PostHog handbook + docs + tutorials frequently restate the same
  procedure at different depths. Dense retrieval alone will surface the wrong depth.
- **Exact-match bait for BM25** — Kubernetes has `CrashLoopBackOff`, `ImagePullBackOff`, `kubectl`
  flags; FastAPI has decorator and parameter names; Zulip has specific setting labels. These are the
  cases where dense-only retrieval loses and hybrid wins, which is the number your case study reports.
- **Built-in staleness traps** — Kubernetes docs carry deprecation and version-skew notices, and the
  FastAPI release notes contradict older tutorial pages. You do not have to fabricate outdated-doc
  traps; find and label the real ones.
- **Genuine no-answer questions** — four unrelated products means plenty of plausible questions the
  corpus simply cannot answer (e.g. asking PostHog billing questions of the Kubernetes docs).

## Metadata extraction notes

- **source_name** — top-level dir under `raw/`
- **section_heading** — markdown headings; PostHog/Zulip files also carry YAML/MDX frontmatter with
  `title`
- **last_updated** — `git -C data/raw/<repo> log -1 --format=%aI -- <relative/path>`
- **doc_type** — derive from path segment (`handbook/`, `tasks/debug/`, `api_docs/`,
  `release-notes.md`, `tutorials/`)
- **access_level** — not present in the data; synthesize it (e.g. handbook → `internal`, product docs
  → `public`) so the metadata-filtering path has something real to filter on

## Reproducing / re-downloading

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git data/raw/fastapi
git -C data/raw/fastapi sparse-checkout set docs/en/docs

git clone --depth 1 --filter=blob:none --sparse https://github.com/kubernetes/website.git data/raw/k8s-website
git -C data/raw/k8s-website sparse-checkout set content/en/docs/tasks content/en/docs/concepts content/en/docs/reference/kubectl

git clone --depth 1 --filter=blob:none --sparse https://github.com/PostHog/posthog.com.git data/raw/posthog
git -C data/raw/posthog sparse-checkout set contents/handbook contents/docs contents/tutorials

git clone --depth 1 --filter=blob:none --sparse https://github.com/zulip/zulip.git data/raw/zulip
git -C data/raw/zulip sparse-checkout set starlight_help/src/content api_docs docs
```

GitLab.com was unreachable from this machine (TLS handshake aborted), so the GitLab Handbook —
otherwise the best single source for policy/onboarding content — was replaced by the PostHog
handbook, which serves the same role.
