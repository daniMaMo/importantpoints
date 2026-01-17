# pips
Structural Points Detection in Time Series
# Structural Points Detection in Time Series

This repository provides Python implementations of two widely used algorithms for identifying structural points in financial time series:

1. **Perceptually Important Points (PIPs)**
2. **Rolling Window (RW) peak and bottom detection**

These methods are designed to extract key points that capture the shape and dynamics of a time series while reducing its overall complexity.

---

## Repository Contents

- `pips.py`  
  Implementation of the **Perceptually Important Points (PIPs)** algorithm, supporting multiple distance metrics:
  - Euclidean Distance (ED)
  - Perpendicular Distance (PD)
  - Vertical Distance (VD)

- `rw.py`  
  Implementation of the **Rolling Window (RW)** method for detecting local peaks and bottoms in a price series.

---

## Implemented Algorithms

### 1. Perceptually Important Points (PIPs)

The PIPs algorithm iteratively identifies the most significant points in a time series. It starts with the first and last observations and, at each iteration, adds the point that maximizes a distance measure with respect to the existing segments.

**Key features:**
- User-defined number of structural points.
- Multiple distance definitions.
- Suitable for:
  - Time series compression
  - Pattern recognition
  - Trend and shape analysis in financial data

---

### 2. Rolling Window (RW)

The Rolling Window method detects regional peaks and bottoms by comparing each point with its neighboring values within a fixed-size window.

**Key features:**
- Window size of \(2w + 1\).
- Explicit identification of local maxima and minima.
- Useful for:
  - Trend change detection
  - Support and resistance analysis
  - Financial data preprocessing

---

## Requirements

- Python ≥ 3.8  
- NumPy  
- Matplotlib
- os
- pandas

Install dependencies using:
```bash
pip install numpy matplotlib pandas
