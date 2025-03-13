# Portfolio Optimization
Co-Authored a university group project on Portfolio optimization

## Overview
This repository contains implementations of various portfolio optimization techniques used in quantitative finance. 
The project focuses on Hierarchical Risk Parity in comparison to other benchmark methodologies to create efficient investment portfolios that stabilise weights that minimise transaction costs.

## Methods
The repository includes several weighting approaches and compares them:
- **Equally Weighted Portfolio**
- **Minimum Variance Portfolio**
- **Risk Parity Portfolio**
- **Hierarchical Risk Parity**

## Data Sources
The optimization models use historical price data from:
- Stocks across major indices
- Index funds
- Currencies
- Commodities

## Results
The empirical results show:
- HRP outperforms in S&P 500 stocks and foreign exchange portfolios
- Mean Variance portfolio performs best in commodities while uniform weighting excels in highly diversified portfolios
- HRP consistently demonstrates lower weight variance than Miinimum Variance, suggesting lower transaction costs
  
## References
1. López de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out-of-Sample"
2. NYU Stern School of Business VLab. EWMA Covariance.
3. Marti, G. (2018). Hierarchical Risk Parity - Implementation & Experiments

## Co-Authors
- Moishe Keselman
- Saptarshi Kumar
- Thomas Horstkamp
