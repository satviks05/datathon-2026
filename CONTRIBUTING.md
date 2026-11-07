# Team workflow

## Ownership

Each person owns the directory and output files assigned in `docs/DATA_CONTRACTS.md`. Work in your assigned directory to minimize merge conflicts. Source CSVs are read-only. Agree on contract changes before implementing them.

Person 4 coordinates integration and changes to shared root files (`README.md`, `.gitignore`, and `docs/`). Other members propose shared-file changes in a separate pull request. Each owner keeps dependencies, run instructions, and their own tests in their work area; avoid editing a shared dependency file.

## Daily workflow

1. Fetch changes and switch to your assigned branch.
2. Merge `origin/main` into your branch before beginning and before requesting review.
3. Commit only files belonging to your task. Use `git status` and `git diff --cached` before committing.
4. Push your branch and open a pull request against `main`.
5. Ask another teammate to review; resolve conflicts on your branch and merge through the pull request.

```bash
git fetch origin
git switch team/311-cleaning
git merge origin/main
# Make your changes, then stage specific paths:
git add pipelines/complaints/
git commit -m "Clean 311 complaints and report coordinate exclusions"
git push -u origin team/311-cleaning
```

Do not push directly to `main`, force-push shared branches, or edit someone else's directory without coordinating. These are team conventions; GitHub branch protection is not configured by this starter.

## Parallel development

The two cleaners can work independently using raw CSVs. Analysis can implement matching against the documented schemas while cleaners are built. The map can use clearly labeled synthetic fixtures stored inside `web/` while analysis is built; never present fixtures as findings. Each owner supplies a small representative fixture in their own work area if needed.

Generated outputs are local build artifacts. Run upstream steps or exchange artifacts outside Git; don't commit competing versions of the same generated file.

## Done means

- Run instructions and dependencies are documented in your directory.
- Outputs conform to the shared contract.
- Counts, missingness, assumptions, and important checks are documented.
- No secrets, virtual environments, or generated artifacts are committed.
- Another teammate can reproduce the result.
