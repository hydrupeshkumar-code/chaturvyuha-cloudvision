from ai.metrics.compute import compute_all_metrics


def calculate_metrics(
    ground_truth,
    prediction
):

    return compute_all_metrics(
        ground_truth,
        prediction
    )