# Package Enrichment

The package registry is maintained in
[laruche](https://github.com/devdanzin/laruche). See laruche's
documentation for:

- The complete field schema (all YAML fields, types, defaults)
- Step-by-step enrichment process
- Troubleshooting common issues
- Claude Code enrichment prompts

## Quick Reference

Fetch the registry:

```bash
labeille registry sync
```

Enrich a package:

1. Clone the package repo, examine its build/test setup
2. Fill in `install_command`, `test_command`, etc. in the YAML
3. Set `enriched: true`
4. Run `labeille registry validate` to check your work

## Registry Batch Operations

labeille provides batch management commands that operate on the
registry directory:

- `labeille registry add-field` — Add a field to all YAML files
- `labeille registry remove-field` — Remove a field
- `labeille registry rename-field` — Rename a field
- `labeille registry set-field` — Set a field's value (with filters)
- `labeille registry validate` — Check YAML against the schema
- `labeille registry rebuild-index` — Rebuild `index.yaml`
- `labeille registry migrate` — Run schema migrations

All commands accept `--registry-dir` to specify a custom location.
Default: `~/.local/share/labeille/registry/`.
