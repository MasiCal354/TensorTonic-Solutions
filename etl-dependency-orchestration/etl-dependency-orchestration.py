def schedule_pipeline(tasks: list, resource_budget: int) -> list:
    """
    Returns a list of schedule dictionaries.
    """
    task_map = {task["name"]: task for task in tasks}
    running = {}
    completed = set()
    schedule = []
    time = 0
    while len(completed) < len(tasks):
        for name in [name for name, end in running.items() if end <= time]:
            completed.add(name)
            del running[name]
        used = sum(task_map[name]["resources"] for name in running)
        ready = sorted((task for task in tasks if task["name"] not in completed and task["name"] not in running and all(dependency in completed for dependency in task["depends_on"])), key=lambda task: task["name"])
        for task in ready:
            if used + task["resources"] <= resource_budget:
                running[task["name"]] = time + task["duration"]
                schedule.append({"task_name": task["name"], "start_time": time})
                used += task["resources"]
        if running:
            time = min(running.values())
    return sorted(schedule, key=lambda item: (item["start_time"], item["task_name"]))
