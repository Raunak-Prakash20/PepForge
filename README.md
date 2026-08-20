# PepForge-AI 🧬
> **Target-Conditioned De Novo Antimicrobial Peptide Design Engine & 5-Stage In Silico Screening Funnel**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/Tests-21%2F21%20Passed-brightgreen.svg)]()

PepForge-AI is an end-to-end computational therapeutics platform for designing potent, non-hemolytic antimicrobial peptides (AMPs) targeting critical multidrug-resistant ESKAPE pathogens (*Pseudomonas aeruginosa*, *Acinetobacter baumannii*, and Methicillin-resistant *Staphylococcus aureus* [MRSA]).

The platform couples a **target-conditioned autoregressive causal transformer decoder** with a rigorous **5-stage in silico screening funnel** grounded in biophysical principles, peptide-specific ADMET models, 3D structural amphipathicity, and patent novelty clearance.

---

## ⚡ 60-Second Quickstart (Try It Immediately)

Get up and running with zero configuration:

```bash
# 1. Clone & install
git clone https://github.com/Raunak-Prakash20/PepForge.git
cd PepForge
pip install -e .

# 2. Run the interactive 60-second guided tour
python demo.py

# 3. Screen any custom peptide sequence
python scripts/run_screening.py --sequence "KIKLLKLLKKAKKLL"

# 4. Generate 20 brand-new peptides for P. aeruginosa
python scripts/run_generate.py --target "Pseudomonas aeruginosa" --num_candidates 20

# 5. Run the verified test suite (21/21 passed)
pytest tests/ -v
```

---

## 🏗️ System Architecture & Engineering Design

PepForge-AI couples a **target-conditioned autoregressive causal transformer decoder** with a **deterministic 5-stage in silico biophysical and ADMET screening funnel**:

### 1. High-Level Architecture Flow

```mermaid
flowchart TD
    subgraph DataEngine["1. Data Ingestion & Clustering"]
        D1["Raw Databases (DBAASP, APD3, UniProt)"] --> D2["Sequence Canonicalization & Sanitizer"]
        D2 --> D3["Greedy CD-HIT 80% Identity Clustering"]
        D3 --> D4["Cluster-Level Zero-Leakage Split (Train 80% / Val 10% / Test 10%)"]
    end

    subgraph GenerativeEngine["2. Conditioned Generative Transformer"]
        D4 --> M1["Conditioning Tokenizer (Target, Gram, Charge, Length)"]
        M1 --> M2["Autoregressive Causal Decoder (6 Layers, 8 Heads, d_model=128)"]
        M2 --> M3["Nucleus Top-p & Repetition Penalty Sampler"]
        M3 --> M4["De Novo Candidate Pool"]
    end

    subgraph ScreeningFunnel["3. 5-Stage In Silico Biophysical & ADMET Funnel"]
        M4 --> S1["Stage 1: bRo5 Physicochemical Gatekeeper (Charge, GRAVY, uH, Boman)"]
        S1 -- Passed --> S2["Stage 2: ADMET Toxicity & Hemolysis (Sliding Patch, HC50, SI)"]
        S2 -- Passed --> S3["Stage 3: ADMET Metabolic Stability (Trypsin/Pepsin, CSI <= 18)"]
        S3 -- Passed --> S4["Stage 4: 3D ESMFold & Helical Structure (pLDDT >= 70, Amphipathicity)"]
        S4 -- Passed --> S5["Stage 5: Freedom-to-Operate Novelty (Levenshtein >= 4, Id <= 75%)"]
        
        S1 -- Reject --> REJ["Triage Diagnostics & Failure Log"]
        S2 -- Reject --> REJ
        S3 -- Reject --> REJ
        S4 -- Reject --> REJ
        S5 -- Reject --> REJ
    end

    subgraph OutputEngine["4. Lead Ranking & Synthesis Advice"]
        S5 --> L1["Multi-Objective Pareto Ranker (Selectivity Index & pLDDT)"]
        L1 --> L2["Synthesis Capping Advisor (N-terminal Ac-, C-terminal -NH2)"]
        L2 --> L3["leads_dossier.json & CLI Rich Dashboard"]
    end

    style DataEngine fill:#1a1d24,stroke:#3b82f6,stroke-width:2px,color:#fff
    style GenerativeEngine fill:#1a1d24,stroke:#10b981,stroke-width:2px,color:#fff
    style ScreeningFunnel fill:#1a1d24,stroke:#f59e0b,stroke-width:2px,color:#fff
    style OutputEngine fill:#1a1d24,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style REJ fill:#371b1d,stroke:#ef4444,stroke-width:1px,color:#fca5a5
```

---

### 2. End-to-End Data Flow & Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as Researcher / CLI
    participant Script as run_generate.py
    participant Tokenizer as PeptideTokenizer
    participant Model as PepForgeGenerator
    participant Funnel as ScreeningFunnel
    participant Filters as 5 Biophysical Filters
    participant FS as File System (results/)

    User->>Script: python scripts/run_generate.py --target "P. aeruginosa" --num_candidates 100
    Script->>Tokenizer: build_condition_prefix(target, gram, charge, len)
    Tokenizer-->>Script: ["[TARGET:P_AERUGINOSA]", "[GRAM:NEGATIVE]", "[CHARGE:MID]", ...]
    Script->>Model: generate_batch(prefix, n=100, top_p=0.90, temp=0.85)
    Model-->>Script: 100 Unique Peptide Sequences

    Script->>Funnel: screen_batch(candidates, target)
    loop For Each Sequence
        Funnel->>Filters: Stage 1 (bRo5 Rules)
        Funnel->>Filters: Stage 2 (Hemolysis Toxicity)
        Funnel->>Filters: Stage 3 (Protease Metabolism)
        Funnel->>Filters: Stage 4 (3D Structure & Amphipathicity)
        Funnel->>Filters: Stage 5 (Novelty Check)
        Filters-->>Funnel: Diagnostic Metrics & Pass/Fail Status
    end

    Funnel-->>Script: List of CandidateEvaluationReports
    Script->>Script: Filter passed leads & sort by Selectivity Index (SI)
    Script->>FS: Save leads_dossier.json
    Script->>User: Render Rich Terminal Dossier Leaderboard
```

---

### 3. The 5-Stage In Silico Screening Funnel (Biophysical Gatekeepers)

Natural and synthetic therapeutic peptides inherently operate in **Beyond Rule of Five (bRo5)** space. PepForge evaluates every candidate through five strict, quantitative biophysical criteria:

```
[ Candidate Peptides (De Novo / Library) ]
                   │
                   ▼
┌────────────────────────────────────────────────────────┐
│ Stage 1: Beyond-Rule-of-5 (bRo5) Biophysical Filter    │
│ • Net charge (+2.0 <= Q <= +6.5 via Henderson-Hasselbalch)│
│ • Hydrophobic ratio (35% <= H_ratio <= 55%)           │
│ • Boman index (< 2.50 kcal/mol)                       │
│ • Eisenberg hydrophobic moment (uH >= 0.350)          │
│ • GRAVY hydropathy (-1.0 <= GRAVY <= +0.5)            │
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: ADMET Toxicity & Hemolysis Model              │
│ • Hydrophobic patch sliding window (max GRAVY <= 1.0)  │
│ • Empirical HC50 prediction (HC50 >= 150.0 ug/mL)      │
│ • Selectivity Index SI = HC50 / MIC (target SI >= 20.0)│
│ • Aromatic ring fraction cutoff (<= 25.0%)             │
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: ADMET Metabolism & Proteolytic Stability      │
│ • Cleavage site mapping: Trypsin, Chymotrypsin, Pepsin │
│ • Cleavage Site Index CSI <= 18.0 sites/100 residues   │
│ • Trypsin recognition site constraint (<= 3 sites)     │
│ • Terminus capping analysis (N-Ac / C-amidation)       │
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 4: 3D Structural Folding & Amphipathicity        │
│ • ESMFold predicted pLDDT confidence (mean >= 70.0)    │
│ • Chou-Fasman secondary structure (helical >= 45.0%)   │
│ • Helical wheel 2D polar projection & segregation      │
└──────────────────────────┬─────────────────────────────┘
                           │ Passed
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 5: Freedom-to-Operate (FTO) & Novelty Filter     │
│ • Sequence identity vs DBAASP/APD3 reference (<= 75%)  │
│ • Levenshtein edit distance constraint (min dist >= 4) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
                 [ Validated Lead Leads ]
```

---

### 4. Key Architectural Decisions

| Architectural Decision | Chosen Strategy | Technical Rationale |
|:---|:---|:---|
| **Exclusion of Lipinski's Rule of 5** | Beyond-Rule-of-5 (bRo5) criteria | Peptides are macromolecules ($\text{MW} > 500\,\text{Da}$). Applying Lipinski's rules to peptides would reject $100\%$ of therapeutic candidates. |
| **Separation of Generator & Verifier** | Adversarial Guardrail Architecture | Neural networks frequently hallucinate non-physical molecules. Decoupling the generator from deterministic mathematical filters guarantees no invalid molecule can pass. |
| **Cluster-Level Splitting** | 80% Greedy CD-HIT Clustering | Random train/test splits in biology cause massive homology leakage (test sequences being 95% identical to train sequences). Splitting by clusters enforces genuine generalization. |
| **Local vs Global Hydropathy** | 9-Residue Sliding Hydrophobic Patch | Global GRAVY fails on peptides with charged tails (like Melittin). A sliding window identifies localized hydrophobic clusters responsible for human erythrocyte lysis. |
| **Terminal Protection Design** | Automated Capping Advice | Synthetic peptides unprotected at their termini are rapidly degraded in human serum. PepForge automatically suggests chemical capping ($N$-Ac, $C$-$\text{NH}_2$). |

---

## 🧪 Benchmark Validation

The screening pipeline was validated against empirical benchmarks to demonstrate discrimination between therapeutically viable AMPs, pore-forming toxins, and non-active sequences:

| Peptide | Origin / Target | Net Charge | Hydrophobic Moment (uH) | Predicted HC50 (ug/mL) | Cleavage Sites | Funnel Status | Primary Triage Rationale |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Magainin-2** | *X. laevis* / P. aeruginosa | +3.03 | 0.286 | 271.8 (Safe) | 8 | Rejected (Stage 1/5) | Sub-threshold uH (0.286 < 0.350); 100% known homolog |
| **Melittin** | *A. mellifera* / Broad Spec | +4.99 | 0.224 | 25.0 (Toxic) | 6 | **Flagged Toxic** (Stage 2) | Hydrophobic patch (1.96) causes severe erythrocyte lysis |
| **LL-37** | Human Cathelicidin | +5.99 | 0.359 | 650.0 (Safe) | 18 | Rejected (Stage 3/5) | High cleavage site frequency (CSI = 48.6 > 18.0) |
| **Indolicidin** | Bovine Neutrophil | +2.99 | 0.190 | 500.0 (Safe) | 7 | Rejected (Stage 1/4) | Non-helical poly-tryptophan conformation |
| **Syn-P18** | Synthetic Derivative | +7.03 | 0.300 | 426.2 (Safe) | 13 | Rejected (Stage 1/5) | Net charge (+7.03) exceeds selectivity envelope (+6.5) |

---

## 🤖 De Novo Generative Model

PepForge implements a lightweight, high-speed **Causal Transformer Decoder** tailored for biopolymer sequences:
* **Vocab Size:** 35 tokens (canonical amino acids, conditioning prefixes, special control tokens).
* **Conditioning Mechanism:** Sequences are generated prepended with conditioning tokens:
  * Pathogen: `<target:P_aeruginosa>`, `<target:A_baumannii>`, `<target:MRSA>`
  * Gram status: `<gram_neg>`, `<gram_pos>`
  * Charge envelope: `<charge_pos>`
  * Target length: `<len_medium>`
* **Sampling Engine:** Top-$p$ (nucleus) sampling ($p = 0.90$), temperature scaling ($T = 0.85$), repetition penalty ($1.15$), and minimum sequence length enforcement.

---

## 🚀 CLI Usage Guide

### 1. Run Interactive 60-Second Demo
```bash
python demo.py
```

### 2. Screen Any Custom Peptide Sequence
```bash
python scripts/run_screening.py --sequence "KIKLLKLLKKAKKLL" --target "Pseudomonas aeruginosa"
```

### 3. Generate De Novo Candidates for a Specific Target
```bash
python scripts/run_generate.py --target "Pseudomonas aeruginosa" --charge 3.5 --length 18 --num_candidates 20
```

### 4. Evaluate Benchmark Reference Library
```bash
python scripts/run_screening.py
```

### 5. Ingest, Clean, and Cluster New AMP Datasets
```bash
python scripts/run_ingest.py --identity 0.80 --split-ratio 0.8 0.1 0.1
```

### 6. Train or Fine-Tune Transformer
```bash
python scripts/run_train.py --epochs 15 --batch-size 32 --lr 5e-4
```

---

## 💻 Python API Usage

You can import PepForge directly into your own computational biology workflows:

```python
from pepforge.pipeline.screening_funnel import ScreeningFunnel

# Initialize the 5-stage funnel
funnel = ScreeningFunnel()

# Screen any candidate sequence
report = funnel.evaluate_candidate(
    sequence="KWKLFKKIPKFLHLAKKF",
    target_organism="Acinetobacter baumannii",
)

print(f"Overall Passed: {report.overall_passed}")
print(f"Net Charge (pH 7.4): {report.bro5_metrics['net_charge_ph74']:+.2f}")
print(f"Predicted HC50 (ug/mL): {report.admet_toxicity_metrics['predicted_hc50_ug_ml']:.1f}")
print(f"Cleavage Sites: {report.admet_metabolism_metrics['total_cleavage_sites']}")
print(f"Synthesis Capping Advice: {report.synthesis_recommendations}")
```

---

## 📂 Repository Map

| Directory / File | Description |
|:---|:---|
| [`demo.py`](demo.py) | **Zero-friction 60-second guided tour** (benchmark triage + live AI generation) |
| [`pepforge/filters/`](pepforge/filters/) | Biophysical gatekeepers (bRo5 rules, hemolysis ADMET, protease stability, 3D folding, FTO) |
| [`pepforge/models/`](pepforge/models/) | Causal transformer decoder, tokenizer with conditioning prefixes, trainer |
| [`pepforge/pipeline/`](pepforge/pipeline/) | Unified `ScreeningFunnel` orchestrator with Pydantic validation |
| [`scripts/`](scripts/) | Ready-to-run terminal entrypoints for generation, screening, training, and ingestion |
| [`models/`](models/) | Serialized model weights checkpoint (`pepforge_model.pt`) |
| [`data/processed/`](data/processed/) | Clustered AMP corpus (80% CD-HIT identity) with zero homology leakage |
| [`results/`](results/) | Benchmark screening evaluation reports and lead candidate dossiers |
| [`tests/`](tests/) | Comprehensive 21-test verification suite (`pytest tests/ -v`) |

---

## 🧪 Test Suite

Run the full suite of 21 unit tests covering all biophysical metrics, ADMET calculators, 3D structural models, and sequence generation:

```bash
pytest tests/ -v
```

---

## 📄 License & Citation

Distributed under the MIT License. See `LICENSE` for more information.

```bibtex
@software{pepforge2026,
  author = {Raunak Prakash},
  title = {PepForge-AI: Target-Conditioned De Novo Antimicrobial Peptide Design Engine},
  year = {2026},
  url = {https://github.com/Raunak-Prakash20/PepForge}
}
```
