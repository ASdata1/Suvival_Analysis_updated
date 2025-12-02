# ==================================
# src/data_gen/generate_features.py

# ==================================

import numpy as np
import pandas as pd


def generate_features(n: int, seed: int | None = None) -> pd.DataFrame:
    """
    Generate synthetic 'pension member' features.

    Narrative:
    - Age: age at study entry (years), typical pre-retirement / early retirement range.
    - Salary: annual salary (log-normal).
    - Tenure: years of contributions in the pension scheme.
    - Gender: 0 = female, 1 = male (for realism; optional in modelling).
    - Risk_score: latent health / financial fragility risk (higher = more likely to claim early).

    Parameters
    ----------
    n : int
        Number of individuals.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ["age", "salary", "tenure", "gender", "risk_score"].
    """
    rng = np.random.default_rng(seed)

    # Age between 55 and 75, slightly skewed older
    age = rng.normal(loc=63, scale=5, size=n)
    age = np.clip(age, 55, 75)

    # Salary in £, log-normal to give a right-skew
    # mode around ~35k–40k, some high earners
    log_salary = rng.normal(loc=10.5, scale=0.35, size=n)
    salary = np.exp(log_salary)  # in £

    # Tenure in years (0–40), correlated with age
    base_tenure = rng.normal(loc=age - 45, scale=5)
    tenure = np.clip(base_tenure, 1, 40)

    # Gender (0/1)
    gender = rng.binomial(1, 0.5, size=n)

    # Latent risk score: higher → more likely to claim pension earlier
    risk_score = rng.normal(loc=0.0, scale=1.0, size=n)

    df = pd.DataFrame(
        {
            "age": age,
            "salary": salary,
            "tenure": tenure,
            "gender": gender,
            "risk_score": risk_score,
        }
    )

    return df
