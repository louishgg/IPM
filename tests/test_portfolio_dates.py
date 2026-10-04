"""Calendar behavior owned by the shared portfolio date utilities."""

from portfolio_core.dates import month_end_index


def test_month_end_index_includes_leap_day():
    assert month_end_index("2024-01-01", "2024-03-31").strftime(
        "%Y-%m-%d"
    ).tolist() == ["2024-01-31", "2024-02-29", "2024-03-31"]
