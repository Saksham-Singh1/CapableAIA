# planner.py
class Planner:
    def __init__(self):
        self.task_queue = []

    def decompose_goal(self, end_goal):
        """Decompose the end goal into smaller actionable tasks."""
        # Simulating task decomposition
        tasks = [
            f"Open browser for '{end_goal}'",
            "Search relevant information",
            "Compile data in a document"
        ]
        self.task_queue.extend(tasks)
        return tasks

    def get_next_task(self):
        """Get the next task from the queue."""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None
