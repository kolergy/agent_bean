import abc

from neo4j   import GraphDatabase
from typing  import List

class Task(abc.ABC):
    """
    Abstract base class for a task.
    """
    def __init__(self, id: int, description: str, inputs: List[str], outputs: List[str], parent_task: Task = None) -> None:
        """
        Initialize a Task.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        :param inputs: The inputs of the task.
        :param outputs: The outputs of the task.
        :param parent_task: The parent task of the task.
        """
        self.id = id
        self.description = description
        self.inputs = inputs
        self.outputs = outputs
        self.parent_task = parent_task

class AtomicTask(Task):
    """
    Class for an atomic task, which is a task that cannot be broken down into smaller tasks.
    """
    def __init__(self, id: int, description: str, action: str, queries: List[str], inputs: List[str], outputs: List[str]) -> None:
        """
        Initialize an AtomicTask.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        :param action: The action to be performed by the task.
        :param queries: The queries associated with the task.
        :param inputs: The inputs of the task.
        :param outputs: The outputs of the task.
        """
        super().__init__(id, description, inputs, outputs)
        self.action = action
        self.queries = queries

class ComplexTask(Task):
    """
    Class for a complex task, which is a task that can be broken down into smaller tasks.
    """
    def __init__(self, id: int, description: str, subtasks: List[Task], inputs: List[str], outputs: List[str], parent_task: Task = None) -> None:
        """
        Initialize a ComplexTask.

        :param id: The unique identifier of the task.
        :param description: The description of the task.
        :param subtasks: The subtasks that make up the complex task.
        :param inputs: The inputs of the task.
        :param outputs: The outputs of the task.
        :param parent_task: The parent task of the complex task.
        """
        super().__init__(id, description, inputs, outputs)
        self.subtasks = subtasks
        self.parent_task = parent_task

class TasksGraph:
    """
    Class for a tasks graph, which represents the tasks and their relationships as a graph using neo4j.
    """
    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initialize a TasksGraph.

        :param uri: The URI of the neo4j database.
        :param user: The user of the neo4j database.
        :param password: The password of the neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        """
        Close the connection to the neo4j database.
        """
        self.driver.close()

    def add_task(self, task: Task) -> None:
        """
        Add a task to the graph.

        :param task: The task to be added.
        """
        with self.driver.session() as session:
            session.run("CREATE (t:Task {id: $id, description: $description, inputs: $inputs, outputs: $outputs})",
                        id=task.id, description=task.description, inputs=task.inputs, outputs=task.outputs)

    def add_relationship(self, task1: Task, task2: Task, relationship: str) -> None:
        """
        Add a relationship between two tasks.

        :param task1: The first task.
        :param task2: The second task.
        :param relationship: The relationship between the tasks.
        """
        with self.driver.session() as session:
            session.run("MATCH (t1:Task {id: $id1}), (t2:Task {id: $id2}) "
                        "CREATE (t1)-[r:" + relationship + "]->(t2)",
                        id1=task1.id, id2=task2.id)
