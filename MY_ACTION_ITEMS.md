# Your Action Items — Milestone 2 Submission

Everything has been built and is ready. Below is exactly what you need to do before submitting.

---

## 1. Create a GitHub Repository

1. Go to [github.com](https://github.com) and sign in
2. Click **New repository**
3. Name it something like `cs57200-15puzzle` or `heuristic-search-15puzzle`
4. Set visibility to **Public** (required so the instructor can access it)
5. Do **not** initialize with a README — you already have one

---

## 2. Push the Code

From inside the `fifteen_puzzle/` folder on your machine, run:

```bash
git init
git add .
git commit -m "Initial commit: baseline + enhancements for Milestone 2"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

The rubric requires **at least 10 commits** showing incremental progress. After the initial push, make a few additional commits as you test and clean things up — for example:

```bash
# Example additional commits
git add src/pdb.py
git commit -m "Add disk caching for PDB tables"

git add data/benchmark.json
git commit -m "Add fixed 100-instance benchmark (seed=42)"

git add results/
git commit -m "Add preliminary results from 10-instance run"
```

---

## 3. Paste the Repo Link into the Report

Open `Milestone_2_Report.docx` and on the first page, add your GitHub URL. The rubric explicitly requires a **code repository link included in the report**.

---

## 4. Run the Test Cases on Your Machine

Before submitting, verify the code runs cleanly on your setup:

```bash
cd fifteen_puzzle/src
python main.py --mode test
```

You should see `✓ SOLVED` for the first four test cases. This satisfies the rubric's "code runs without errors" and "at least 3 test cases demonstrated" requirements.

---

## 5. Submit on Brightspace

The rubric requires two things submitted:
- The **Word report** (`Milestone_2_Report.docx`)
- The **repository link** (pasted inside the report or as a submission comment)

**Due date: April 9, 2026**

---

## 6. For the Final Submission (Weeks 15–16)

When you come back to finish the project, the main remaining tasks are:

- Run the full 100-instance benchmark: `python main.py --mode benchmark --n 100`
- Try the 7-8 PDB partition for stronger results: `--partition 7-8` (needs ~400–600 MB RAM and 30–60 min to build)
- Insert the 4 generated figures (`results/figures/`) into the final report
- Expand results with statistical analysis (Wilcoxon signed-rank tests are already described in the code)
- Grow the report to 10–15 pages per the final rubric
- Prepare a 15-minute presentation
