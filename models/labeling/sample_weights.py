"""
Sample weighting schemes
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 4
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class SampleWeights:
    """
    Sample weighting schemes from López de Prado Ch. 4

    Addresses the challenge of overlapping labels in financial data:
    - Sequential bootstrap for backtesting
    - Sample uniqueness (overlap adjustment)
    - Return attribution weighting
    - Time decay weighting
    """

    @staticmethod
    def get_sample_uniqueness(labels: pd.DataFrame,
                             molecule: Optional[pd.Index] = None) -> pd.Series:
        """
        Calculate sample uniqueness (overlap adjustment)

        This addresses the problem of overlapping labels where multiple
        predictions share the same return observations.

        Args:
            labels: DataFrame with 'first_touch_time' column
            molecule: Subset of labels to calculate uniqueness for

        Returns:
            Series of uniqueness weights
        """
        if molecule is None:
            molecule = labels.index

        logger.info(f"Calculating sample uniqueness for {len(molecule)} samples")

        overlaps = pd.DataFrame(index=molecule)

        for idx in molecule:
            start_time = idx
            end_time = labels.loc[idx, 'first_touch_time']

            if pd.isna(end_time):
                overlaps.loc[idx, 'overlap'] = 0
                continue

            # Count overlapping labels
            overlap_mask = (
                ((labels.index >= start_time) & (labels.index <= end_time)) |
                ((labels['first_touch_time'] >= start_time) &
                 (labels['first_touch_time'] <= end_time))
            )

            overlaps.loc[idx, 'overlap'] = overlap_mask.sum()

        # Calculate uniqueness as inverse of overlap
        uniqueness = 1 / (1 + overlaps['overlap'])

        logger.info(f"Mean uniqueness: {uniqueness.mean():.4f}")

        return uniqueness

    @staticmethod
    def get_return_attribution(labels: pd.DataFrame) -> pd.Series:
        """
        Weight samples by their return contribution

        Samples with larger absolute returns get higher weights,
        as they contain more information.

        Args:
            labels: DataFrame with 'return' column

        Returns:
            Series of return-based weights
        """
        returns = labels['return'].abs()

        # Avoid division by zero
        if returns.sum() == 0:
            logger.warning("Zero total returns, using uniform weights")
            return pd.Series(1.0 / len(returns), index=labels.index)

        weights = returns / returns.sum()

        logger.info(f"Return attribution weights: mean={weights.mean():.4f}, max={weights.max():.4f}")

        return weights

    @staticmethod
    def get_time_decay(labels: pd.DataFrame, decay_factor: float = 0.95) -> pd.Series:
        """
        Apply time decay to older samples

        More recent observations get higher weights as they may be
        more relevant to current market conditions.

        Args:
            labels: DataFrame with index as timestamps
            decay_factor: Exponential decay factor (0 < decay_factor < 1)

        Returns:
            Series of time-decay weights
        """
        if not isinstance(labels.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, cannot apply time decay")
            return pd.Series(1.0, index=labels.index)

        # Calculate days from most recent
        latest = labels.index.max()
        days_old = (latest - labels.index).days

        # Apply exponential decay
        weights = decay_factor ** days_old

        # Normalize
        weights = weights / weights.sum()

        logger.info(f"Time decay weights: mean={weights.mean():.4f}, oldest={weights.min():.4f}")

        return pd.Series(weights, index=labels.index)

    @staticmethod
    def combine_weights(labels: pd.DataFrame,
                       use_uniqueness: bool = True,
                       use_return: bool = True,
                       use_time_decay: bool = False,
                       decay_factor: float = 0.95) -> pd.Series:
        """
        Combine multiple weighting schemes

        Args:
            labels: DataFrame with labels and returns
            use_uniqueness: Include uniqueness weighting
            use_return: Include return attribution weighting
            use_time_decay: Include time decay weighting
            decay_factor: Decay factor for time decay

        Returns:
            Combined weight series
        """
        logger.info("Combining sample weights")

        weights_list = []

        if use_uniqueness:
            uniqueness = SampleWeights.get_sample_uniqueness(labels)
            weights_list.append(uniqueness)

        if use_return:
            return_weights = SampleWeights.get_return_attribution(labels)
            weights_list.append(return_weights)

        if use_time_decay:
            time_weights = SampleWeights.get_time_decay(labels, decay_factor)
            weights_list.append(time_weights)

        if not weights_list:
            logger.warning("No weights specified, using uniform weights")
            return pd.Series(1.0 / len(labels), index=labels.index)

        # Average the weights
        combined = pd.DataFrame(weights_list).T.mean(axis=1)

        # Normalize
        combined = combined / combined.sum()

        logger.info(f"Combined weights: mean={combined.mean():.4f}, std={combined.std():.4f}")

        return combined

    @staticmethod
    def sequential_bootstrap(labels: pd.DataFrame,
                           sample_weights: pd.Series,
                           n_samples: Optional[int] = None) -> pd.Index:
        """
        Sequential bootstrap accounting for label overlap

        Standard bootstrap doesn't account for overlapping labels.
        Sequential bootstrap adjusts selection probabilities based on
        how much each sample overlaps with already selected samples.

        Args:
            labels: DataFrame with 'first_touch_time' column
            sample_weights: Sample weights
            n_samples: Number of samples to draw (default: same as input)

        Returns:
            Index of selected samples
        """
        if n_samples is None:
            n_samples = len(labels)

        logger.info(f"Sequential bootstrap: drawing {n_samples} samples")

        selected = []
        available_idx = labels.index.tolist()
        current_weights = sample_weights.copy()

        for i in range(n_samples):
            if len(available_idx) == 0:
                break

            # Normalize weights
            prob = current_weights[available_idx]
            prob = prob / prob.sum()

            # Sample one index
            sampled_idx = np.random.choice(available_idx, p=prob)
            selected.append(sampled_idx)

            # Update weights based on overlap with sampled index
            if i < n_samples - 1:  # Don't need to update on last iteration
                sampled_start = sampled_idx
                sampled_end = labels.loc[sampled_idx, 'first_touch_time']

                if not pd.isna(sampled_end):
                    # Reduce weights of overlapping samples
                    for idx in available_idx:
                        if idx == sampled_idx:
                            continue

                        idx_start = idx
                        idx_end = labels.loc[idx, 'first_touch_time']

                        if pd.isna(idx_end):
                            continue

                        # Check for overlap
                        overlap = (
                            (idx_start >= sampled_start and idx_start <= sampled_end) or
                            (idx_end >= sampled_start and idx_end <= sampled_end) or
                            (sampled_start >= idx_start and sampled_start <= idx_end)
                        )

                        if overlap:
                            # Reduce weight by uniqueness factor
                            current_weights[idx] *= 0.5

        logger.info(f"Sequential bootstrap complete: selected {len(selected)} samples")

        return pd.Index(selected)
