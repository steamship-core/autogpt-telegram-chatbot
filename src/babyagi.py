from collections import deque
from typing import Dict, Any, List, Optional

from langchain import LLMChain
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field
from steamship import Steamship
from steamship_langchain.llms import OpenAIChat
from steamship_langchain.vectorstores import SteamshipVectorStore

from chains import TaskCreationChain, TaskPrioritizationChain
from prompts import get_prompt, get_tools


def get_next_task(
    task_creation_chain: LLMChain,
    result: Dict,
    task_description: str,
    task_list: List[str],
    objective: str,
) -> List[Dict]:
    """Get the next task."""
    incomplete_tasks = ", ".join(task_list)
    response = task_creation_chain.run(
        result=result,
        task_description=task_description,
        incomplete_tasks=incomplete_tasks,
        objective=objective,
    )
    new_tasks = response.split("\n")
    return [{"task_name": task_name} for task_name in new_tasks if task_name.strip()]


def prioritize_tasks(
    task_prioritization_chain: LLMChain,
    this_task_id: int,
    task_list: List[Dict],
    objective: str,
) -> List[Dict]:
    """Prioritize tasks."""
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    response = task_prioritization_chain.run(
        task_names=task_names, next_task_id=next_task_id, objective=objective
    )
    new_tasks = response.split("\n")
    prioritized_task_list = []
    for task_string in new_tasks:
        if not task_string.strip():
            continue
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            prioritized_task_list.append({"task_id": task_id, "task_name": task_name})
    return prioritized_task_list


def _get_top_tasks(vectorstore, query: str, k: int) -> List[str]:
    """Get the top k tasks based on the query."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    if not results:
        return []
    sorted_results, _ = zip(*sorted(results, key=lambda x: x[1], reverse=True))
    return [str(item.metadata["task"]) for item in sorted_results]


def execute_task(
    vectorstore, execution_chain: LLMChain, objective: str, task: str, k: int = 5
) -> str:
    """Execute a task."""
    context = _get_top_tasks(vectorstore, query=objective, k=k)
    return execution_chain.run(objective=objective, context=context, task=task)


class BabyAGI(Chain, BaseModel):
    """Controller model for the BabyAGI agent."""

    task_list: deque = Field(default_factory=deque)
    task_creation_chain: TaskCreationChain = Field(...)
    task_prioritization_chain: TaskPrioritizationChain = Field(...)
    execution_chain: AgentExecutor = Field(...)
    task_id_counter: int = Field(1)
    vectorstore: VectorStore = Field(init=False)
    max_iterations: Optional[int] = None

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def print_task_list(self):
        yield "\n*****TASK LIST*****\n"
        for t in self.task_list:
            yield str(t["task_id"]) + ": " + t["task_name"]

    def print_next_task(self, task: Dict):
        yield "\n*****NEXT TASK*****\n"
        yield str(task["task_id"]) + ": " + task["task_name"]

    def print_task_result(self, result: str):
        yield "\n*****TASK RESULT*****\n"
        yield result

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent."""
        objective = inputs["objective"]
        first_task = inputs.get("first_task", "Make a todo list")
        self.add_task({"task_id": 1, "task_name": first_task})
        num_iters = 0
        while True:
            if self.task_list:
                yield from self.print_task_list()

                # Step 1: Pull the first task
                task = self.task_list.popleft()
                yield from self.print_next_task(task)

                # Step 2: Execute the task
                result = execute_task(
                    self.vectorstore, self.execution_chain, objective, task["task_name"]
                )
                this_task_id = int(task["task_id"])
                yield from self.print_task_result(result)

                # Step 3: Store the result in the VectorStore
                result_id = f"result_{task['task_id']}"
                self.vectorstore.add_texts(
                    texts=[result],
                    metadatas=[{"task": task["task_name"]}],
                    ids=[result_id],
                )

                # Step 4: Create new tasks and reprioritize task list
                new_tasks = get_next_task(
                    self.task_creation_chain,
                    result,
                    task["task_name"],
                    [t["task_name"] for t in self.task_list],
                    objective,
                )
                for new_task in new_tasks:
                    self.task_id_counter += 1
                    new_task.update({"task_id": self.task_id_counter})
                    self.add_task(new_task)
                self.task_list = deque(
                    prioritize_tasks(
                        self.task_prioritization_chain,
                        this_task_id,
                        list(self.task_list),
                        objective,
                    )
                )
            num_iters += 1
            if self.max_iterations is not None and num_iters == self.max_iterations:
                yield "\n*****TASK ENDING*****\n"
                break

    @classmethod
    def from_llm(
        cls,
        client: Steamship,
        llm: BaseLLM,
        vectorstore: VectorStore,
        verbose: bool = False,
        **kwargs,
    ) -> "BabyAGI":
        """Initialize the BabyAGI Controller."""
        tools = get_tools(client, **kwargs)
        prompt = get_prompt(tools)
        task_creation_chain = TaskCreationChain.from_llm(llm, verbose=verbose)
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm, verbose=verbose
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True
        )
        return cls(
            task_creation_chain=task_creation_chain,
            task_prioritization_chain=task_prioritization_chain,
            execution_chain=agent_executor,
            vectorstore=vectorstore,
            **kwargs,
        )


def solve_agi_problem(
    client,
    objective,
    model_name: str,
    max_tokens: int,
    max_iterations: Optional[int] = None,
):
    llm = OpenAIChat(
        client=client, temperature=0, model_name=model_name, max_tokens=max_tokens
    )
    vectorstore = SteamshipVectorStore(
        client=client,
        index_name=f"{client.config.workspace_handle}_index_{hash(objective)}",
        embedding="text-embedding-ada-002",
    )

    # Logging of LLMChains
    verbose = True
    # If None, will keep on going forever
    iterations: Optional[int] = max_iterations if max_iterations > 0 else None
    baby_agi = BabyAGI.from_llm(
        client=client,
        llm=llm,
        model_name=model_name,
        vectorstore=vectorstore,
        verbose=verbose,
        max_iterations=iterations,
        max_tokens=max_tokens,
    )

    yield from baby_agi._call({"objective": objective})


if __name__ == "__main__":
    client = Steamship(workspace="agi_tools_pro")
    for k in solve_agi_problem(
        client,
        "Write status report on Andrew Tate",
        max_tokens=256,
        max_iterations=3,
        model_name="gpt-3.5-turbo",
    ):
        print("HELLO", k)
