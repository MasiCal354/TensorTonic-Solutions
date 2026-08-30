def retraining_policy(daily_stats: list, config: dict) -> list:
    """
    Returns a list of retraining day numbers.
    """
    retrain_days = []
    budget = config["budget"]
    last_retrain_day = -config["cooldown"]
    days_since_retrain = 0
    for stats in daily_stats:
        days_since_retrain += 1
        triggered = (stats["drift_score"] > config["drift_threshold"] or stats["performance"] < config["performance_threshold"] or days_since_retrain >= config["max_staleness"])
        allowed = stats["day"] - last_retrain_day >= config["cooldown"] and budget >= config["retrain_cost"]
        if triggered and allowed:
            retrain_days.append(stats["day"])
            budget -= config["retrain_cost"]
            last_retrain_day = stats["day"]
            days_since_retrain = 0
    return retrain_days
