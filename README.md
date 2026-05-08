# Copula Risk Modeling — Canadian P&C Insurance Market

**Author:** Reda Hakkani | PhD Candidate, Applied Mathematics | Montréal, QC  
**Domain:** Actuarial Science · Quantitative Risk · Canadian P&C Insurance  
**Regulatory context:** OSFI MCT · IBC Calibration · Solvency II

---

## Overview

Models the **dependency structure between catastrophe losses and liability claims** in the Canadian P&C insurance market using four copula families: Clayton, Gumbel, Frank, and Gaussian.

**Canadian market calibration:**
- CAT losses: Alberta hail, BC wildfire, Quebec ice storm, Ontario flood
- Liability: Bodily Injury + Property Damage across personal auto and commercial lines
- Latent correlation ρ = 0.62 (consistent with IBC catastrophe co-occurrence studies)
- Regulatory framework: OSFI Minimum Capital Test (MCT), OSFI A-4 Guideline

---

## Key Results

| Metric | Value |
|--------|-------|
| Observations | 1,200 |
| Kendall's τ | 0.4283 |
| Spearman ρ | 0.6058 |
| **Best Copula (AIC/BIC)** | **Gumbel** |
| λ_L Clayton (Lower tail) | 0.4930 |
| λ_U Gumbel (Upper tail) | 0.1455 |
| VaR 75% (Aggregate) | CAD 0.199B |
| VaR 90% | CAD 0.209B |
| **VaR 99.5% (OSFI MCT)** | **CAD 0.233B** |
| Monte-Carlo Simulations | 500 |

### AIC / BIC Model Comparison

| Copula | θ / ρ | AIC | BIC | Rank |
|--------|-------|-----|-----|------|
| **Gumbel** | 1.1223 | **-7,791** | **-7,786** | **#1** |
| Gaussian | 0.6039 | -559 | -554 | #2 |
| Frank | 4.3807 | -528 | -523 | #3 |
| Clayton | 0.9800 | -444 | -439 | #4 |

---

## Copula Families & Canadian Market Interpretation

### Gumbel Copula — Best Fit *(AIC/BIC)*
- Upper tail dependence (λ_U > 0): captures simultaneous extreme high-value losses
- Relevant for Canadian multi-peril years (e.g., 2020: Alberta hail + BC wildfire)
- Recommended for reinsurance XL pricing and CAT treaty structures

### Clayton Copula — Strong Lower Tail
- Lower tail dependence λ_L = 0.493: co-occurrence of moderate-to-large losses
- Critical for OSFI stress-testing: simultaneous CAT + liability deterioration
- **OSFI implication:** Gaussian copula underestimates tail risk (no tail dependence)

### Frank Copula
- Symmetric, no tail dependence
- Appropriate for moderate inter-line correlation scenarios

### Gaussian Copula *(industry standard — insufficient for CAT)*
- Zero tail dependence — underestimates concurrent extreme losses
- Included as benchmark; not recommended for OSFI MCT capital calculation

---

## Methodology

```
Canadian P&C Loss Data (CAT + Liability)
                │
                ▼
    Marginal Fitting — Log-Normal
    (IBC calibrated: μ_CAT=13.2, σ=0.90)
                │
                ▼
    Probability Integral Transform
    → Uniform margins [0,1]
                │
                ▼
    MLE Parameter Estimation
    ├── Clayton  (θ > 0)
    ├── Gumbel   (θ ≥ 1)
    ├── Frank    (θ ∈ ℝ)
    └── Gaussian (ρ ∈ (-1,1))
                │
                ▼
    AIC / BIC Model Selection
                │
                ▼
    Tail Dependence λ_L / λ_U
                │
                ▼
    Monte-Carlo Validation (500 runs)
    → VaR 75% / 90% / 99.5% (OSFI MCT)
```

---

## Regulatory Framework

| Standard | Requirement | This Model |
|----------|-------------|------------|
| **OSFI MCT** | Capital at 99.5% VaR | ✓ Computed |
| **OSFI A-4** | Dependency assumptions | ✓ Copula-based |
| **Solvency II** | SCR correlation | ✓ Equivalent methodology |
| **IBC** | Industry loss calibration | ✓ Applied |
| **CIA** | Actuarial standards | ✓ Aligned |

---

## Installation

```bash
git clone https://github.com/RedaHakkani/copula-risk-modeling.git
cd copula-risk-modeling
pip install -r requirements.txt
python src/copula_modeling.py
```

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Output

- Full console report (dependency measures, AIC/BIC, tail dependence, VaR matrix)
- `copula_risk_analysis.png` — 6-panel professional dashboard

---

## References

- Joe, H. (1997). *Multivariate Models and Dependence Concepts*. Chapman & Hall.
- Nelsen, R.B. (2006). *An Introduction to Copulas*. Springer.
- IBC (2023). *Facts of the General Insurance Industry in Canada*.
- OSFI (2022). *Guideline A-4 — Regulatory Capital and Internal Capital Targets*.
- EIOPA (2014). *Solvency II Technical Specifications*.

---

*Reda Hakkani — PhD Candidate, Applied Mathematics | Montréal, QC*  
*Available for actuarial and quantitative risk roles — hakkanireda@hotmail.com*
