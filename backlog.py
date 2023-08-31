import abc

class Task(abc.ABC):
    def __init__(self, id, description):
        self.id = id
        self.description = description

class AtomicTask(Task):
    def __init__(self, id, description, action, queries):
        super().__init__(id, description)
        self.action = action
        self.queries = queries

class ComplexTask(Task):
    def __init__(self, id, description, subtasks):
        super().__init__(id, description)
        self.subtasks = subtasks

class Backlog:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)
