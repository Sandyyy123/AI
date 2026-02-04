## Working with the notebook in VS Code

1. Open the notebook: `31st_Jan_1st_Feb_2026_clean.ipynb`.
2. Ensure the correct Python environment is selected (VS Code status bar).
3. Run cells with the ▶️ buttons or use **Run All**.
4. To keep diffs clean, consider **Clear All Outputs** before committing.

## Commit and push notebook changes

```bash
# Check status
git status -sb

# Stage the notebook (and any other files you changed)
git add 31st_Jan_1st_Feb_2026_clean.ipynb README.md

# Commit
git commit -m "Update notebook and documentation"

# Push to the current branch
git push origin HEAD
```
