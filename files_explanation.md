# FILES EXPLANATION
This file contains an overview of what is performed by each of the files in the repository. Only the relevant files have been included

- `alpha_regression.py`: The returns of the pairs trading strategy are regressed quarterly in order to obtain the quarterly alphas of the strategy's factor regression, which will later be regressed on the company data in `factor_regression_company_data.ipynb`.
- `factor_creation_msci_4f.py`: Creates the mkt-rf, smb, hml and mom factors from the different MSCI datasets.
- `factor_creation.ipynb`: Creates the factors for the SP500 dataset
- `factor_regression_company_data.ipynb`: The quarterly alphas obtained in `alpha_regression_v3.py` are regressed here on the company data. Used for SP500.
- `factor_regression_esg_msci.py`: Regresses msci stocks excess returns on the 4 created factors in `factor_creation_msci_4f.py`, IEP and ESG score for monthly values.
- `factor_regression_msci.py`: Regresses msci stocks excess returns on the 4 created factors in `factor_creation_msci_4f.py`and IEP
- `factor_regression_v5_latex.py`: Creates the regressions used in the written latex document for the primary factor regression.
- `factor_regression_v5.py`: Creates the primary regression for the sp500
- `iep_democracy_msci_regression.py`: Try at creating a regression on the democracy score. However, it does not currently work due to the country code being inconsistent in the stock ticker and the democracy dataset
- `iep_esg_msci_regression.py`: Regresses the company IEP on the esg score, currently with no company control factors
- `johansen_portfolio_log_v8.py`: Calculates the returns of the johansen portfolio on the sp500
- `performance_analysis.ipynb`: Calculates several metrics regarding the performance of the pairs trading strategy
- `SPX_MLE_v11.py`: Calculates the returns of the pairs trading strategy on the sp500
- `volatility_calculations_msci.py`: Calculates the IEP on the msci dataset
- `volatility_calculations.ipynb`: Calculates the IEP on the sp500 dataset and several other volatility related factors. 
- `tfg-escrito/sections/test.ipynb`: Creates the IEP mean +- 2 standard deviations area plot