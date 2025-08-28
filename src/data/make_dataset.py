# Create a small synthetic churn dataset.

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N = 3000

# Feature (simple, mostly numeric for a clean demo)
tenure = np.random.gamma(shape=2.0, scale=12.0, size=N).clip(0, 72)
monthly = np.random.normal(loc=70, scale=25, size=N).clip(10, 200)
total = (tenure * monthly * np.random.uniform(0.8, 1.2, size=N)).clip(100, 12000)
support_calls = np.random.poisson(lam=2, size=N).clip(0, 15)
contracts_left = np.random.randint(0, 12, size=N)
is_senior = np.random.binomial(1, p=0.2, size=N)

# Hidden true rule to create churn probability
logit = (
    -1.0
    + 0.015 * (200 - monthly)
    + 0.02 * (5 - support_calls)
    - 0.02 * tenure
    + 0.1 * is_senior
    + 0.03 * (3 - contracts_left)
)
prob = 1 / (1 + np.exp(-logit))

churn = np.random.binomial(1, prob)

os.makedirs('data', exist_ok=True)

df = pd.DataFrame(
    {
        'tenure_months': tenure.round(1),
        'monthly_charges': monthly.round(2),
        'total_charges': total.round(2),
        'support_calls': support_calls,
        'contracts_left': contracts_left,
        'is_senior': is_senior,
        'churn': churn,
    }
)

csv_path = os.path.join('data', 'churn.csv')
df.to_csv(csv_path, index=False)
print(f'Wrote {csv_path} with shape {df.shape}')
