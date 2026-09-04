# Contributing

Read `AGENTS.md`, `docs/ARCHITECTURE.md`, and `docs/VERIFICATION.md` before changing code.

Install the locked development environment and run the full gate:

```bash
uv sync --locked --group dev
make check
```

Every test must carry exactly one primary marker: `unit`, `architecture`, `contract`,
`integration`, or `scientific`. Environment requirements such as `requires_elmer` are additional
markers, not primary layers.

Markdown equations must render on GitHub: use `$...$` inline and `$$...$$` for display math. Avoid
backslash math delimiters, equation environments, `\operatorname`, and document-local TeX macros.
Run `make markdown` to check the complete documentation tree.

Changes to a public schema, backend protocol, package boundary, execution policy, result authority,
or scientific claim require an ADR. A backend capability may be advertised only after an associated
conformance and scientific validation test exists.
