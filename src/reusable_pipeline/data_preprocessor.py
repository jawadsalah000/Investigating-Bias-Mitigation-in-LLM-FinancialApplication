"""HMDA data preprocessing utilities for the reusable pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Load, clean, and transform the HMDA slice used across scenarios."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: pd.DataFrame | None = None

    def DataLoader(self) -> "DataPreprocessor":
        self.data = pd.read_csv(self.path, low_memory=False)
        return self

    def DataFilter(self) -> "DataPreprocessor":
        if self.data is None:
            raise ValueError("Data not loaded. Call DataLoader first.")

        d = self.data.copy()
        d = d[d["derived_loan_product_type"] == "Conventional:First Lien"]
        d = d[d["derived_dwelling_category"] == "Single Family (1-4 Units):Site-Built"]
        d = d[d["conforming_loan_limit"] == "C"]
        d = d[d["loan_purpose"] == 1]
        d = d[d["reverse_mortgage"] == 2]
        d = d[d["open-end_line_of_credit"] == 2]
        d = d[d["business_or_commercial_purpose"] == 2]
        d = d[d["occupancy_type"] == 1]
        d = d[d["derived_sex"] == "Male"]
        d = d[d["derived_race"] == "White"]
        d = d[d["loan_term"] == "360"]
        d = d[d["action_taken"].isin([1, 3])]
        d.loc[d["action_taken"] == 3, "action_taken"] = 2

        drop_cols = [
            "activity_year",
            "lei",
            "lien_status",
            "purchaser_type",
            "derived_msa-md",
            "loan_type",
            "state_code",
            "county_code",
            "census_tract",
            "derived_ethnicity",
            "preapproval",
            "total_points_and_fees",
            "origination_charges",
            "discount_points",
            "lender_credits",
            "prepayment_penalty_term",
            "intro_rate_period",
            "negative_amortization",
            "interest_only_payment",
            "balloon_payment",
            "other_nonamortizing_features",
            "construction_method",
            "manufactured_home_secured_property_type",
            "manufactured_home_land_property_interest",
            "total_units",
            "multifamily_affordable_units",
            "applicant_credit_score_type",
            "co-applicant_credit_score_type",
            "applicant_ethnicity-1",
            "applicant_ethnicity-2",
            "applicant_ethnicity-3",
            "applicant_ethnicity-4",
            "applicant_ethnicity-5",
            "co-applicant_ethnicity-1",
            "co-applicant_ethnicity-2",
            "co-applicant_ethnicity-3",
            "co-applicant_ethnicity-4",
            "co-applicant_ethnicity-5",
            "applicant_ethnicity_observed",
            "co-applicant_ethnicity_observed",
            "applicant_race-1",
            "applicant_race-2",
            "applicant_race-3",
            "applicant_race-4",
            "applicant_race-5",
            "co-applicant_race-1",
            "co-applicant_race-2",
            "co-applicant_race-3",
            "co-applicant_race-4",
            "co-applicant_race-5",
            "applicant_race_observed",
            "co-applicant_race_observed",
            "applicant_sex",
            "co-applicant_sex",
            "applicant_sex_observed",
            "co-applicant_sex_observed",
            "applicant_age_above_62",
            "co-applicant_age",
            "co-applicant_age_above_62",
            "submission_of_application",
            "initially_payable_to_institution",
            "aus-1",
            "aus-2",
            "aus-3",
            "aus-4",
            "aus-5",
            "denial_reason-1",
            "denial_reason-2",
            "denial_reason-3",
            "denial_reason-4",
            "tract_population",
            "tract_minority_population_percent",
            "ffiec_msa_md_median_family_income",
            "tract_to_msa_income_percentage",
            "tract_owner_occupied_units",
            "tract_one_to_four_family_homes",
            "tract_median_age_of_housing_units",
            "interest_rate",
            "total_loan_costs",
            "rate_spread",
        ]
        self.data = d.drop(columns=[c for c in drop_cols if c in d.columns], errors="ignore")
        return self

    def DataCleaner(self) -> "DataPreprocessor":
        if self.data is None:
            raise ValueError("Data not loaded. Call DataLoader first.")

        critical = [
            "loan_amount",
            "loan_to_value_ratio",
            "loan_term",
            "income",
            "debt_to_income_ratio",
            "derived_race",
            "derived_sex",
            "loan_purpose",
            "occupancy_type",
            "property_value",
            "applicant_age",
        ]
        self.data = self.data.dropna(subset=[c for c in critical if c in self.data.columns])
        return self

    def DataTransformer(self) -> "DataPreprocessor":
        if self.data is None:
            raise ValueError("Data not loaded. Call DataLoader first.")

        d = self.data.copy()

        def extract_average(age: object) -> float | np.nan:
            if pd.isnull(age):
                return np.nan
            s = str(age)
            if "-" in s:
                a, b = s.split("-")
                return (float(a) + float(b)) / 2
            if "<" in s:
                return float(s.replace("<", "")) - 1
            if ">" in s:
                return float(s.replace(">", "")) + 1
            try:
                return float(s)
            except Exception:
                return np.nan

        def clean_range(val: object) -> float | np.nan:
            if pd.isnull(val):
                return np.nan
            s = str(val).replace("%", "").replace("<", "").replace(">", "").strip()
            try:
                if "-" in s:
                    a, b = s.split("-")
                    return (float(a.strip()) + float(b.strip())) / 2
                return float(s)
            except Exception:
                return np.nan

        if "applicant_age" in d.columns:
            d["applicant_age"] = d["applicant_age"].apply(extract_average)
        if "debt_to_income_ratio" in d.columns:
            d["debt_to_income_ratio"] = d["debt_to_income_ratio"].apply(clean_range)

        for col in ["income", "loan_amount", "loan_to_value_ratio", "debt_to_income_ratio", "property_value"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce")

        if "income" in d.columns:
            d["income"] = d["income"] * 1000

        self.data = d
        return self

    def remove_outliers_iqr(self) -> "DataPreprocessor":
        if self.data is None:
            raise ValueError("Data not loaded. Call DataLoader first.")

        d = self.data.copy()
        features = ["income", "loan_amount", "loan_to_value_ratio", "debt_to_income_ratio", "property_value"]
        for f in features:
            if f not in d.columns:
                continue
            q1, q3 = d[f].quantile(0.25), d[f].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            d = d[(d[f] >= lower) & (d[f] <= upper)]

        d = d.reset_index(drop=True)
        drop_final = [
            "conforming_loan_limit",
            "derived_loan_product_type",
            "derived_dwelling_category",
            "reverse_mortgage",
            "loan_purpose",
            "derived_sex",
            "loan_term",
            "occupancy_type",
            "hoepa_status",
            "business_or_commercial_purpose",
            "index",
            "open-end_line_of_credit",
        ]
        d = d.drop(columns=[c for c in drop_final if c in d.columns], errors="ignore")
        self.data = d
        return self

    def add_unique_ids(self) -> "DataPreprocessor":
        if self.data is None:
            raise ValueError("Data not loaded. Call DataLoader first.")

        length = len(self.data.index)
        self.data["applicant_id"] = [f"APP-{i + 1:05d}" for i in range(length)]
        return self

    def run(self) -> pd.DataFrame:
        return (
            self.DataLoader()
            .DataFilter()
            .DataCleaner()
            .DataTransformer()
            .remove_outliers_iqr()
            .add_unique_ids()
        ).data.copy()
