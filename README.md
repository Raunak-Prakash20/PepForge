# PepForge-AI 🧬
> **Target-Conditioned De Novo Antimicrobial Peptide Design Engine**

PepForge-AI is an end-to-end computational therapeutics platform for designing potent, non-hemolytic antimicrobial peptides (AMPs) targeting critical multidrug-resistant pathogens (*Pseudomonas aeruginosa*, *Acinetobacter baumannii*, and MRSA).

---

## 🛠️ The 5-Stage In Silico Screening Funnel

1. **Stage 1 (bRo5 Physicochemical Rules):** Beyond-Rule-of-5 criteria for peptides (Net charge $+2.0 \le Q \le +6.5$, Hydrophobic ratio $35\% \le H_{ratio} \le 55\%$, Boman Index $< 2.50\text{ kcal/mol}$, Hydrophobic Moment $\mu_H \ge 0.35$).
2. **Stage 2 (ADMET Toxicity):** In silico hemolysis predictor ($HC_{50} \ge 150\,\mu\text{g/mL}$, Selectivity Index $SI = HC_{50} / MIC \ge 20.0$).
3. **Stage 3 (ADMET Metabolism):** Endopeptidase cleavage mapper (Trypsin, Chymotrypsin, Pepsin) and terminal capping recommendations ($N$-acetylation, $C$-amidation).
4. **Stage 4 (3D Structure & Amphipathicity):** ESMFold structural verification (Mean $\text{pLDDT} \ge 70.0$, alpha-helical facial segregation).
5. **Stage 5 (Novelty & Homology Check):** Homology clearance against DBAASP/APD3 (Sequence Identity $\le 75\%$).

---

## 🚀 Quickstart

```bash
# Install in editable mode
pip install -e .

# Run the 5-Stage Screening Funnel Demo
python scripts/run_screening.py

# Run the test suite
pytest tests/ -v
```
