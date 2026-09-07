# CS6886 Assignment 2 — LaTeX report

Single-column, Overleaf-ready source for *MobileNet-v2 Compression on CIFAR-10:
Uniform Quantisation and Sensitivity-Guided Mixed Precision*.

```
report/
├── main.tex                  # the report (article, 10pt, single column)
├── references.bib            # BibTeX: HAWQ-V2, Hutchinson, CIFAR-10, SGDR, MobileNet-v2
├── make_figures.py           # regenerates every figure from repository artefacts
├── main.pdf                  # compiled reference PDF (8 pages)
├── tables/results_tables.tex # numeric tables, \input by main.tex
└── figures/                  # .pdf (used by LaTeX) + .png for each figure
```

Everything `main.tex` needs is inside this directory; no path outside `report/`
is referenced.

## 1. Compiling locally

```bash
cd report
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

or `latexmk -pdf main.tex` / `tectonic -X compile main.tex`.

Packages used are all standard TeX Live: `geometry`, `amsmath`, `graphicx`,
`booktabs`, `colortbl`, `xcolor`, `caption`, `microtype`, `hyperref`,
`titlesec`, `hyphenat`, `lmodern`. No custom class or style file.

## 2. Uploading to Overleaf

```bash
cd ..
zip -r report.zip report
```

In Overleaf: **New Project → Upload Project → report.zip**. `main.tex` is
detected as the main document; compiler is **pdfLaTeX**. Figures are included
without a file extension, so LaTeX picks the vector `.pdf` and the `.png`
copies serve as previews.

## 3. Provenance of each figure

Run `python report/make_figures.py` from the repository root to regenerate all
of them. No value is drawn by hand or transcribed.

| Figure | File | Source |
|---|---|---|
| 1 | `augmentation_examples` | montage of existing `runs/augmented_data_q1/*.png` (from `src/visualize_augmentations.py`) |
| 2 | `loss_curve` | `runs/baseline/history.csv` (`train_loss`, `test_loss`) |
| 3 | `accuracy_curve` | `runs/baseline/history.csv` (`train_acc`, `test_acc`) |
| 4 | `confusion_matrix` | `runs/baseline/best.pt` (epoch 176) evaluated over all 10,000 test images; reproduces 93.18% top-1 and the published per-class accuracies exactly. Cells give test-image counts; with 1,000 images per class the diagonal divided by ten is the per-class accuracy |
| 5 | `q2_uniform_sweep` | `runs/compression/w{32a32,8a8,8a4,4a8,4a4,2a8,2a2}.json` |
| 6 | `hawq_sensitivity` | `runs/mixed_precision/sensitivity.json` (`ranking`, `hawq_score`) |
| 7 | `wandb_parallel_coordinates` | copied verbatim from `docs/wandb.png`, the exported W&B parallel-coordinates panel |

The confusion-matrix cell counts are cached in
`figures/confusion_counts.npy`; delete that file to force re-evaluation of the
checkpoint (needs `conda activate thala`, torch 2.14.0 and a GPU). Every other
figure needs only `numpy` and `matplotlib` plus the recorded CSV/JSON files.

Table values come from `runs/compression/*.json`,
`runs/mixed_precision/results.csv` and `docs/Q4_answers.md`. The two compression
conventions are kept distinct and separately labelled, as in the source
artefacts: Table 5 reports the whole-model `total_compression_ratio` logged for
each sweep run, while Table 6 reports the final storage audit (7.86× weights,
4.00× activations, 6.21× whole model).

## 4. Notes and outstanding TODOs

1. **Name / Roll Number / GitHub URL** — ruled blanks in the `\author{}` block of
   `main.tex`. The repository has no git remote, so no URL was invented.
2. **W&B link** — the sweep ran with `--wandb-mode offline`, so the runs live in
   `./wandb/offline-run-*` and have no public URL. Figure 7 is the exported
   panel and the report says so explicitly; no dashboard link is claimed. To
   publish them, run `wandb sync wandb/offline-run-*`.
3. **`docs/q3_parallel_coordinates_top9.png` is not used.** That local
   reproduction of the same nine configurations carries the title
   *"VGG16: FP32 baseline + 4 uniform + 4 best mixed-precision configs"*, which
   mislabels the model (this work is MobileNet-v2). The W&B export
   `docs/wandb.png` carries no such title and is used instead. If the local
   version is preferred, fix its title before including it.

No accuracy, compression ratio, confusion-matrix entry, curve or sweep value in
the report is a placeholder.
