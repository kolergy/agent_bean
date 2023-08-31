import abc

class Task(abc.ABC):
    """
    Abstract base class for a task.
    """
    def __init__(self, id: int, description: str) -> None:
        """
        Initialize a Task.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        """
        self.id = id
        self.description = description

class AtomicTask(Task):
    """
    Class for an atomic task, which is a task that cannot be broken down into smaller tasks.
    """
    def __init__(self, id: int, description: str, action: str, queries: List[str]) -> None:
        """
        Initialize an AtomicTask.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        :param action: The action to be performed by the task.
        :param queries: The queries associated with the task.
        """
        super().__init__(id, description)
        self.action = action
        self.queries = queries

class ComplexTask(Task):
    """
    Class for a complex task, which is a task that can be broken down into smaller tasks.
    """
    def __init__(self, id: int, description: str, subtasks: List[Task]) -> None:
        """
        Initialize a ComplexTask.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        :param subtasks: The subtasks that make up the complex task.
        """
        super().__init__(id, description)
        self.subtasks = subtasks

class Backlog:
    """
    Class for a backlog, which is a list of tasks to be completed.
    """
    def __init__(self) -> None:
        """
        Initialize a Backlog.
        """
        self.tasks = []

    def add_task(self, task: Task) -> None:
        """
        Add a task to the backlog.

        :param task: The task to be added.
        """
        self.tasks.append(task)
