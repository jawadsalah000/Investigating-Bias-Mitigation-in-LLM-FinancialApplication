import numpy as np
import pandas as pd


class StratifiedSampler:
    """Selects a cohort with roughly even coverage across key numerical features."""

    def __init__(self, features, sample_size, n_bins=4, random_state=42):
        self.features = features
        self.sample_size = sample_size
        self.n_bins = n_bins
        self.random_state = random_state

    def sample(self, df, filters=None):
        filtered = self._apply_filters(df, filters).copy()
        if filtered.empty:
            raise ValueError("No rows available after applying filters for stratified sampling.")

        # Bin each numeric feature into quantiles so we can treat the combination of bins as strata.
        binned = self._add_bins(filtered)
        buckets = self._build_buckets(binned)

        # Pull one index from each bucket in round-robin fashion to keep coverage even.
        chosen_indices = self._round_robin_sample(buckets)

        if len(chosen_indices) < self.sample_size:
            shortfall = self.sample_size - len(chosen_indices)
            remaining = filtered.drop(index=chosen_indices)
            if len(remaining) < shortfall:
                raise ValueError("Not enough rows to satisfy requested sample size.")
            rng = np.random.default_rng(self.random_state)
            extra_indices = rng.choice(remaining.index.to_numpy(), size=shortfall, replace=False)
            chosen_indices.extend(extra_indices.tolist())

        result = filtered.loc[chosen_indices].copy()
        result = result.drop(columns=[c for c in result.columns if c.endswith('_bin')], errors='ignore')
        # Shuffle order is handled upstream; here we just strip helper columns and re-index.
        return result.reset_index(drop=True)

    def _apply_filters(self, df, filters):
        if not filters:
            return df
        result = df
        for column, value in filters.items():
            if callable(value):
                result = result[value(result[column])]
            elif isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
                result = result[result[column].isin(value)]
            else:
                result = result[result[column] == value]
        return result

    def _add_bins(self, df):
        working = df.copy()
        for feature in self.features:
            if feature not in working.columns:
                continue
            try:
                working[f"{feature}_bin"] = pd.qcut(
                    working[feature],
                    q=self.n_bins,
                    duplicates='drop'
                )
            except ValueError:
                # Fallback: create a single bin if feature has too few unique values
                working[f"{feature}_bin"] = pd.cut(
                    working[feature],
                    bins=min(self.n_bins, working[feature].nunique()),
                    include_lowest=True
                )
        bin_cols = [c for c in working.columns if c.endswith('_bin')]
        if bin_cols:
            working['bin_combination'] = working[bin_cols].astype(str).agg('_'.join, axis=1)
        else:
            working['bin_combination'] = 'all'
        return working

    def _build_buckets(self, binned_df):
        buckets = {}
        for combo, group in binned_df.groupby('bin_combination'):
            buckets[combo] = group.index.tolist()
        return buckets

    def _round_robin_sample(self, buckets):
        rng = np.random.default_rng(self.random_state)
        groups = list(buckets.keys())
        rng.shuffle(groups)

        selected = []
        step_groups = {g: list(indices) for g, indices in buckets.items() if indices}

        while len(selected) < self.sample_size and any(step_groups.values()):
            for group in list(step_groups.keys()):
                indices = step_groups[group]
                if not indices:
                    continue
                idx_position = rng.integers(0, len(indices))
                selected.append(indices.pop(idx_position))
                if len(selected) >= self.sample_size:
                    break
                if not indices:
                    step_groups.pop(group, None)
        return selected
