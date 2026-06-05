# Stochastic Interest Rate Modelling and Prediction
### Cox-Ingersoll-Ross (CIR) Model | Finance Club, IIT Roorkee — Open Projects 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MlwKQL8p7cgoEiQtFu3qM8H11QbRKH-i?usp=sharing)

---

## Project Overview

This project implements, calibrates, and extends the **Cox-Ingersoll-Ross (CIR)** short-rate model on real historical zero-coupon bond yield data. The core challenge is reconstructing the **full yield curve (6M through 2Y)** using **only the 3-Month yield** as input on any given test day — a single-factor term structure prediction problem grounded in stochastic calculus.

> **Evaluation Metric:** Out-of-sample R² computed by flattening all maturities and dates into a single array → `r2_score(y_true.flatten(), y_pred.flatten())`  
> **Threshold:** R² > 0.85 (Finance Club verification requirement)

---

## Results

| Model | Out-of-Sample R² | Threshold | Status |
|---|---|---|---|
| Base CIR — Cross-Section MLE | **0.8909** | 0.85 |  PASS |
| Two-Factor CIR (Longstaff-Schwartz 1992) | **0.865** | 0.85 |  PASS |

### Per-Maturity Breakdown — Base CIR

| Maturity | R² | RMSE (bps) | Observation |
|---|---|---|---|
| 6M | 0.9939 | ~5 | Excellent — very close to 3M input |
| 9M | 0.9654 | ~13 | Very good — affine structure holds well |
| 1Y | 0.9061 | ~21 | Good — CIR captures level changes |
| 2Y | 0.3864 | ~40 | Hardest — inverted curve in test period |

> **Why is 2Y hardest?** The test period (Apr 2024 – Apr 2026) features an **inverted yield curve** where 3M rates exceed 2Y rates. The single-factor CIR model, calibrated on training data where the curve was mostly upward sloping, struggles to capture this structural inversion consistently.

---

## Repository Structure

```text
 Stochastic Interest Rate Modelling and
Prediction/
├── README.md                           # Project documentation
└── Finance_Club_23118037_Kanishka_Gupta.ipynb   # Complete CIR Interest Rate Modelling notebook

```
---

## Dataset

| Property | Detail |
|---|---|
| Maturities | 3M, 6M, 9M, 1Y, 2Y, 5Y, 10Y, 20Y, 30Y |
| Training period | 19 May 2016 → 26 Apr 2024 |
| Training observations | 1,976 daily rows × 9 maturity columns |
| Test period | 29 Apr 2024 → 29 Apr 2026 |
| Test observations | 495 daily rows |
| Test input | **3M yield only** — all other maturities are held out |
| Asset class | Undisclosed (model relies purely on mathematical relationships) |

---

## Mathematical Framework

### 1. The CIR Stochastic Differential Equation

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

| Parameter | Symbol | Calibrated Value | Meaning |
|---|---|---|---|
| Mean-reversion speed | κ | **0.1689** | Half-life of rate shocks ≈ **4.1 years** |
| Long-run mean | θ | **0.0272** (2.72%) | Equilibrium interest rate |
| Volatility coefficient | σ | **0.1029** | Size of stochastic shocks |

The **square-root diffusion** $\sigma\sqrt{r_t}$ ensures:
- Volatility is proportional to the rate level (empirically realistic)
- Rates stay non-negative when the **Feller condition** $2\kappa\theta \geq \sigma^2$ holds

**Feller check:** $2 \times 0.1689 \times 0.0272 = 0.00919 < \sigma^2 = 0.01059$ — slightly violated at calibrated values. This is handled in Monte Carlo via a reflection scheme and has no practical impact since test-period rates remain far from zero.

---

### 2. Closed-Form Zero-Coupon Bond Price

$$P(t, T) = A(\tau)\,e^{-B(\tau)\,r_t}, \qquad \tau = T - t$$

$$B(\tau) = \frac{2(e^{\gamma\tau}-1)}{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}$$

$$A(\tau) = \left[\frac{2\gamma\,e^{(\kappa+\gamma)\tau/2}}{(\gamma+\kappa)(e^{\gamma\tau}-1)+2\gamma}\right]^{\!\dfrac{2\kappa\theta}{\sigma^2}}$$

where $\gamma = \sqrt{\kappa^2 + 2\sigma^2}$ is the auxiliary parameter.

---

### 3. Yield Formula — Affine in $r_t$

$$y(t, \tau) = -\frac{\ln P(t,T)}{\tau} = \underbrace{\frac{B(\tau)}{\tau}}_{\text{slope in }r_t} \cdot r_t - \underbrace{\frac{\ln A(\tau)}{\tau}}_{\text{intercept}}$$

This **affine structure** is the key: the entire yield curve is a linear function of today's short rate. Given calibrated $(\kappa, \theta, \sigma)$, the full curve is reconstructed from the 3M rate alone in a single closed-form evaluation — no simulation, no iteration.

---

## Calibration: Cross-Sectional Weighted MLE

### Method

Parameters are estimated by minimising the weighted cross-sectional yield error across all training observations and all 8 maturities:

$$(\hat\kappa,\,\hat\theta,\,\hat\sigma) = \arg\min_{\kappa,\theta,\sigma} \sum_{t=1}^{1976} \sum_{j=1}^{8} w_j \bigl[y^{obs}(t,\tau_j) - y^{CIR}(r_{0,t},\,\tau_j;\,\kappa,\theta,\sigma)\bigr]^2$$

### Maturity Weights

All 8 training maturities (6M through 30Y) are included — longer maturities carry structural information about $\kappa$ and $\sigma$ that short maturities alone cannot identify:

| Maturity | 6M | 9M | 1Y | 2Y | 5Y | 10Y | 20Y | 30Y |
|---|---|---|---|---|---|---|---|---|
| Weight $w_j$ | 2.0 | 2.0 | 2.0 | 2.0 | 1.0 | 0.5 | 0.5 | 0.5 |

Weights are higher for 6M–2Y because these are the test-set prediction maturities.

### Why Cross-Sectional Rather Than Time-Series MLE?

| Method | Optimises | Aligns With Prediction Task? |
|---|---|---|
| Time-series MLE | Likelihood of $r_t \to r_{t+dt}$ transitions | ❌ No — targets dynamics, not curve shape |
| OLS on SDE | Regression of $\Delta r_t$ on $r_t$ | ❌ No — same issue, often gives negative $\kappa$ |
| **Cross-section MLE (used)** | Yield curve reconstruction error | ✅ Yes — directly minimises the evaluation objective |

### Optimisation

L-BFGS-B with **120 restarts** (6×5×4 grid on $\kappa_0, \theta_0, \sigma_0$ starting values) to ensure convergence to the global minimum rather than a local one.

---

## Extension: Two-Factor CIR (Longstaff-Schwartz 1992)

### Motivation

The base CIR model has a fundamental constraint: it is a **one-factor model**. The yield curve shape is entirely determined by $r_t$ — it cannot independently move the **level** and **slope** of the curve. In the test period, the yield curve is inverted, which one factor alone cannot capture consistently.

### Model Structure

The short rate is decomposed into two independent CIR factors:

$$r_t = X_t + Y_t$$

$$dX_t = \kappa_X(\theta_X - X_t)\,dt + \sigma_X\sqrt{X_t}\,dW_t^X \quad \text{(level factor, slow)}$$

$$dY_t = \kappa_Y(\theta_Y - Y_t)\,dt + \sigma_Y\sqrt{Y_t}\,dW_t^Y \quad \text{(slope factor, fast)}$$

with $\langle dW^X, dW^Y \rangle = 0$ (factors are independent).

### Bond Pricing

By independence, the bond price factorises:

$$P(t,T) = A_X(\tau)\,A_Y(\tau)\,\exp\!\bigl(-B_X(\tau)\,X_0 - B_Y(\tau)\,Y_0\bigr)$$

where $A_i(\tau)$, $B_i(\tau)$ are the standard CIR functions applied to each factor's parameters.

### Identification from 3M Rate

Since only $r_0 = X_0 + Y_0$ is observed:

$$X_0 = \alpha\,r_0, \qquad Y_0 = (1-\alpha)\,r_0, \qquad \alpha \in (0,1)$$

The split parameter $\alpha$ is jointly estimated with the six factor parameters $(\kappa_X, \theta_X, \sigma_X, \kappa_Y, \theta_Y, \sigma_Y)$ during calibration.

### Result

The Two-Factor CIR achieves **out-of-sample R² ≈ 0.865**, passing the verification threshold and providing richer yield curve dynamics through the level-slope decomposition.

---

**Zero look-ahead bias** — no test-period yield data enters any estimation step.

---

 
## How to Run

1. Click the **"Open in Colab"** badge at the top of this README or **[click here](https://colab.research.google.com/drive/1MlwKQL8p7cgoEiQtFu3qM8H11QbRKH-i?usp=sharing)** to open the notebook directly
2. Go to **Runtime → Run all**
3. When Cell 5 runs, a **"Choose Files"** button appears — upload:
   - `train_data.csv`
   - `test_data.csv`
   - `test_data_3M.csv`
4. The notebook completes fully in **3–5 minutes** on a standard Colab CPU

> No additional `pip install` needed — all dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`) are pre-installed in Google Colab.
---
## Critical Analysis Summary

### Where the Model Succeeds
- **6M and 9M yields:** Near-perfect prediction (R² > 0.96) because these maturities are tightly coupled to the 3M rate via the CIR affine structure
- **1Y yield:** Good prediction (R² = 0.91) as the mean-reversion effect is still small at this horizon

### Where the Model Fails
- **2Y yield (R² = 0.39):** The test period features an inverted curve (3M > 2Y on average). CIR produces inversion only when $r_t > \theta$, but the calibrated $\theta = 2.72\%$ means that when $r_t \approx 4.9\%$, the model predicts yields that *increase* with maturity rather than *decrease*

### Fundamental Limitations

| Limitation | Consequence |
|---|---|
| **One observable input** | All predictions are a deterministic function of the 3M rate alone |
| **Single factor** | Cannot independently capture level, slope, and curvature movements |
| **Constant parameters** | Cannot adapt to structural regime shifts (e.g., rate hiking cycle) |
| **No jumps** | Sudden ±75 bps policy decisions are not modelled |
| **Risk-neutral calibration** | Mixes risk-neutral and real-world dynamics |

### What κ = 0.169 Tells Us

$$\text{Shock half-life} = \frac{\ln 2}{\kappa} \approx 4.1 \text{ years}$$

Rate deviations from equilibrium take approximately **4 years** to halve — consistent with multi-year monetary policy cycles observed in the training data (near-zero rates 2016–2022, then rapid tightening 2022–2024).

---

## Submission Details

- **Project:** Finance Club, IIT Roorkee — Open Projects 2026
- **Type:** Single-Person Project | Duration: 2 Weeks
- **Notebook:** Google Colab (link at top — sharing set to "Anyone with the link can view")
- **Evaluation:** Out-of-sample R² > 0.85 on flattened yield predictions
