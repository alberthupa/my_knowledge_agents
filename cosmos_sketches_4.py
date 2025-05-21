import os
from collections import defaultdict, Counter
import datetime
from collections import Counter, defaultdict
import json  # For potentially loading mock data or saving results

from src.vectors.cosmos_client import SimpleCosmosClient


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
DATABASE_NAME = "hupi-loch"
PARTITION_KEY_PATH = "/id"

# --- Configuration ---
KEYWORD_FIELDS = [
    "keywords",
    "companies",
    "model_name",
    "model_architecture",
    "detailed_model_version",
    "ai_tools",
    "infrastructure",
    "ml_techniques",
]

# Define time windows in days (relative to 'today', which is the execution day)
# We'll typically analyze data *up to* 'yesterday'.
TIME_WINDOWS = {
    "daily": 1,  # Data from yesterday
    "weekly": 7,  # Data from the last 7 days (yesterday to 7 days ago)
    "monthly": 30,  # Data from the last 30 days
    "quarterly": 90,  # Data from the last 90 days
}

SIGNIFICANT_INCREASE_THRESHOLD = 2.0  # Factor by which current count must exceed baseline average to be "significant"
TOP_N_FOR_PERIOD_SUMMARY = 5  # Number of top items to show for each period's summary
TOP_N_FOR_EMERGING_TREND_DETAILS = (
    10  # Max number of items to show in flattened emerging trend sections
)


cosmos_client = SimpleCosmosClient(
    connection_string=COSMOS_CONNECTION_STRING,
    database_name=DATABASE_NAME,
    partition_key_path=PARTITION_KEY_PATH,
)

cosmos_client.connect()
container = cosmos_client.database_client.get_container_client("knowledge-pieces")


def get_mock_items_for_date_range(start_date_str, end_date_str):
    # print(f"MOCK FETCH: Querying items from {start_date_str} to {end_date_str}")
    query = f"SELECT c.id, c.chunk_date, {', '.join(['c.' + f for f in KEYWORD_FIELDS])} FROM c WHERE c.chunk_date >= '{start_date_str}' AND c.chunk_date <= '{end_date_str}'"
    # query = f"SELECT c.id, c.chunk_date, {', '.join(['c.' + f for f in KEYWORD_FIELDS])} FROM c"
    # print(query)
    items = list(container.query_items(query=query, enable_cross_partition_query=True))
    print(f"Found {len(items)} items.")
    return items


# --- Date Utilities ---
def get_date_ranges(today_date):
    """
    Calculates the start and end dates for each time window relative to today_date.
    Analysis is typically done up to 'yesterday'.
    Returns a dictionary: {"window_name": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
    """
    ranges = {}
    yesterday = today_date - datetime.timedelta(days=1)

    for window_name, days_delta in TIME_WINDOWS.items():
        start_date = yesterday - datetime.timedelta(days=days_delta - 1)
        ranges[window_name] = {
            "start": start_date.strftime("%Y-%m-%d"),
            "end": yesterday.strftime("%Y-%m-%d"),
        }
    return ranges


# --- Data Processing and Aggregation ---
def aggregate_keyword_counts(items):
    """
    Aggregates counts of each keyword from the provided items.
    Returns a dictionary: {"field_name": Counter_object}
    """
    aggregated_counts = defaultdict(Counter)
    if not items:
        return aggregated_counts

    for item in items:
        for field in KEYWORD_FIELDS:
            if field in item and isinstance(item[field], list):
                for keyword in item[field]:
                    if keyword:
                        aggregated_counts[field][keyword.strip()] += 1
    return aggregated_counts


# --- Trend Analysis ---
def identify_top_n_keywords(period_counts, n=10):
    """
    Identifies the top N keywords for each field in a given period's counts.
    Returns: {"field_name": [("keyword", count), ...]}
    """
    top_keywords = {}
    for field, counts in period_counts.items():
        top_keywords[field] = counts.most_common(n)
    return top_keywords


def compare_trends(
    current_period_counts,
    adjusted_previous_period_average_counts,
    significant_increase_threshold=2.0,
):
    """
    Compares current period counts to adjusted average counts from a previous baseline.
    Returns:
        "newly_emerging": {"field_name": [("keyword", current_count), ...]}
        "significantly_increased": {"field_name": [("keyword", current_count, prev_avg_count, increase_factor), ...]}
    """
    newly_emerging = defaultdict(list)
    significantly_increased = defaultdict(list)

    for field in KEYWORD_FIELDS:
        current_field_counts = current_period_counts.get(field, Counter())

        for keyword, current_count in current_field_counts.items():
            previous_avg_count = adjusted_previous_period_average_counts.get(
                field, Counter()
            ).get(keyword, 0)

            if previous_avg_count < 0.01 and current_count > 0:
                newly_emerging[field].append((keyword, current_count))
            elif (
                previous_avg_count > 0
                and current_count / previous_avg_count >= significant_increase_threshold
            ):
                increase_factor = current_count / previous_avg_count
                significantly_increased[field].append(
                    (
                        keyword,
                        current_count,
                        round(previous_avg_count, 2),
                        round(increase_factor, 2),
                    )
                )

        newly_emerging[field].sort(key=lambda x: x[1], reverse=True)
        significantly_increased[field].sort(
            key=lambda x: (x[3], x[1]), reverse=True
        )  # Sort by factor, then current_count

    return {
        "newly_emerging": newly_emerging,
        "significantly_increased": significantly_increased,
    }


def calculate_adjusted_baseline_counts(
    current_period_counts,
    baseline_period_counts,
    days_current_period,
    days_baseline_period,
):
    """
    Calculates the average keyword counts for a period equivalent to the 'current_period'
    derived from the 'baseline_period', excluding the contribution of the 'current_period' itself.
    """
    adjusted_avg_counts = defaultdict(Counter)

    if days_baseline_period <= days_current_period:
        # This case should be handled before calling, but as a safeguard
        return adjusted_avg_counts

    days_in_remainder_of_baseline = days_baseline_period - days_current_period
    if days_in_remainder_of_baseline <= 0:
        return adjusted_avg_counts  # No remainder to calculate average from

    num_equivalent_current_periods_in_remainder = (
        days_in_remainder_of_baseline / days_current_period
    )
    if (
        num_equivalent_current_periods_in_remainder <= 0
    ):  # Should not happen if days_in_remainder > 0
        return adjusted_avg_counts

    for field in KEYWORD_FIELDS:
        all_keywords_in_field_baseline = baseline_period_counts.get(
            field, Counter()
        ).keys()
        for keyword in all_keywords_in_field_baseline:
            current_kw_count_in_current_period = current_period_counts.get(
                field, Counter()
            ).get(keyword, 0)
            baseline_kw_count_overall = baseline_period_counts.get(
                field, Counter()
            ).get(keyword, 0)

            # Count of keyword in the baseline period EXCLUDING the current period's contribution
            count_in_remainder = max(
                0, baseline_kw_count_overall - current_kw_count_in_current_period
            )

            avg_kw_count_in_equivalent_prior_period = (
                count_in_remainder / num_equivalent_current_periods_in_remainder
            )
            adjusted_avg_counts[field][keyword] = (
                avg_kw_count_in_equivalent_prior_period
            )

    return adjusted_avg_counts


# --- Main Orchestration ---
def main():
    execution_date = datetime.date.today()
    print(
        f"Trend Analysis Report for data up to: {(execution_date - datetime.timedelta(days=1)).strftime('%Y-%m-%d')}\n"
    )

    date_ranges_for_windows = get_date_ranges(execution_date)
    all_window_counts = {}

    # 1. Fetch and Aggregate data for each time window
    print("--- Period Summaries ---")
    for window_name, dates in date_ranges_for_windows.items():
        print(
            f"\nProcessing: {window_name.upper()} (From {dates['start']} to {dates['end']})"
        )
        items_for_window = get_mock_items_for_date_range(dates["start"], dates["end"])

        if not items_for_window:
            print(f"No items found for {window_name} window.")
            all_window_counts[window_name] = defaultdict(Counter)
            continue

        aggregated_counts = aggregate_keyword_counts(items_for_window)
        all_window_counts[window_name] = aggregated_counts

        top_n = identify_top_n_keywords(aggregated_counts, n=TOP_N_FOR_PERIOD_SUMMARY)
        print(f"Top {TOP_N_FOR_PERIOD_SUMMARY} keywords for {window_name}:")
        has_any_top_keywords = False
        for field, top_list in top_n.items():
            if top_list:
                has_any_top_keywords = True
                print(f"  {field.replace('_', ' ').title()}: {top_list}")
        if not has_any_top_keywords:
            print("  No prominent keywords found for any field in this period.")
    print("\n" + "=" * 60 + "\n")

    # 2. Perform and Print Emerging Trend Comparisons
    trend_comparison_configs = [
        {
            "name": "Daily Emerging Trends",
            "current_period": "daily",
            "baseline_period": "weekly",
        },
        {
            "name": "Weekly Emerging Trends",
            "current_period": "weekly",
            "baseline_period": "monthly",
        },
        {
            "name": "Monthly Emerging Trends",
            "current_period": "monthly",
            "baseline_period": "quarterly",
        },
    ]

    for config in trend_comparison_configs:
        print(f"--- {config['name']} ---")
        current_win = config["current_period"]
        baseline_win = config["baseline_period"]

        if (
            current_win not in all_window_counts
            or baseline_win not in all_window_counts
            or not all_window_counts[current_win]
            or not all_window_counts[baseline_win]
        ):  # check if counters are not empty
            print(
                f"Insufficient data for {current_win} or {baseline_win} period. Skipping {config['name']}.\n"
            )
            continue

        current_counts = all_window_counts[current_win]
        baseline_counts = all_window_counts[baseline_win]
        days_current = TIME_WINDOWS[current_win]
        days_baseline = TIME_WINDOWS[baseline_win]

        if days_baseline <= days_current:
            print(
                f"Baseline window ('{baseline_win}') must be longer than current window ('{current_win}'). Skipping.\n"
            )
            continue

        adjusted_baseline = calculate_adjusted_baseline_counts(
            current_counts, baseline_counts, days_current, days_baseline
        )

        if not adjusted_baseline:  # Check if baseline calculation yielded results
            print(
                f"Could not calculate a valid adjusted baseline from '{baseline_win}' for '{current_win}'. Skipping.\n"
            )
            continue

        period_trends = compare_trends(
            current_counts, adjusted_baseline, SIGNIFICANT_INCREASE_THRESHOLD
        )

        # Flatten and print "Newly Appearing"
        all_newly_emerging = []
        for field, kws_data in period_trends["newly_emerging"].items():
            for kw, count in kws_data:
                all_newly_emerging.append(
                    {"keyword": kw, "field": field, "count": count}
                )

        all_newly_emerging.sort(key=lambda x: x["count"], reverse=True)

        print("\n  Newly Appearing Keywords (vs. avg. of prior periods from baseline):")
        if not all_newly_emerging:
            print("    No newly appearing keywords found.")
        else:
            for item in all_newly_emerging[:TOP_N_FOR_EMERGING_TREND_DETAILS]:
                print(
                    f"    - {item['keyword']} (from {item['field'].replace('_',' ').title()}) - Count: {item['count']}"
                )
            if len(all_newly_emerging) > TOP_N_FOR_EMERGING_TREND_DETAILS:
                print(
                    f"    ... and {len(all_newly_emerging) - TOP_N_FOR_EMERGING_TREND_DETAILS} more."
                )

        # Flatten and print "Rapidly Growing"
        all_significantly_increased = []
        for field, kws_data in period_trends["significantly_increased"].items():
            for kw, cur_count, prev_avg, factor in kws_data:
                all_significantly_increased.append(
                    {
                        "keyword": kw,
                        "field": field,
                        "current_count": cur_count,
                        "prev_avg": prev_avg,
                        "factor": factor,
                    }
                )

        # Sort by factor (desc), then current_count (desc)
        all_significantly_increased.sort(
            key=lambda x: (x["factor"], x["current_count"]), reverse=True
        )

        print(
            f"\n  Rapidly Growing Keywords (Factor >= {SIGNIFICANT_INCREASE_THRESHOLD}x, vs. avg. of prior periods from baseline):"
        )
        if not all_significantly_increased:
            print("    No rapidly growing keywords found.")
        else:
            for item in all_significantly_increased[:TOP_N_FOR_EMERGING_TREND_DETAILS]:
                print(
                    f"    - {item['keyword']} (from {item['field'].replace('_',' ').title()}) - Now: {item['current_count']}, Avg. Prior: {item['prev_avg']:.2f}, Factor: {item['factor']:.2f}x"
                )
            if len(all_significantly_increased) > TOP_N_FOR_EMERGING_TREND_DETAILS:
                print(
                    f"    ... and {len(all_significantly_increased) - TOP_N_FOR_EMERGING_TREND_DETAILS} more."
                )
        print("\n" + "=" * 60 + "\n")

    print("Trend analysis complete.")


if __name__ == "__main__":
    main()
