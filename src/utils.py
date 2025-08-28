from dataclasses import dataclass


@dataclass
class FeatureInfo:
    numeric = [
        'tenure_months',
        'monthly_charges',
        'total_charges',
        'support_calls',
        'contracts_left',
        'is_senior'
    ]