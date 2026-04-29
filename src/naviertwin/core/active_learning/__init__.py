"""Active learning acquisition and query public API."""

from naviertwin.core.active_learning.acquisition import (
    expected_improvement as acquisition_expected_improvement,
)
from naviertwin.core.active_learning.acquisition import (
    greedy_batch_acquisition,
    lower_confidence_bound,
    max_variance_query,
    probability_of_improvement,
    query_by_committee,
    thompson_sample,
    upper_confidence_bound,
)
from naviertwin.core.active_learning.query import (
    expected_improvement as query_expected_improvement,
)
from naviertwin.core.active_learning.query import greedy_maxmin, top_variance_query

expected_improvement = acquisition_expected_improvement

__all__ = [
    "acquisition_expected_improvement",
    "expected_improvement",
    "greedy_batch_acquisition",
    "greedy_maxmin",
    "lower_confidence_bound",
    "max_variance_query",
    "probability_of_improvement",
    "query_by_committee",
    "query_expected_improvement",
    "thompson_sample",
    "top_variance_query",
    "upper_confidence_bound",
]
