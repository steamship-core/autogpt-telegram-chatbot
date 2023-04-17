from langchain import PromptTemplate, LLMChain
from langchain.agents import ZeroShotAgent, Tool
from steamship_langchain import OpenAI
from steamship_langchain.tools import SteamshipSERP


def get_tools(client):
    todo_prompt = PromptTemplate.from_template(
        "You are a planner who is an expert at coming up with a todo list for a given objective. Come up with a todo list for this objective: {objective}"
    )
    todo_chain = LLMChain(llm=OpenAI(client=client, temperature=0), prompt=todo_prompt)
    search = SteamshipSERP(client=client)
    return [
        Tool(
            name="Search",
            func=search.search,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="TODO",
            func=todo_chain.run,
            description="useful for when you need to come up with todo lists. Input: an objective to create a todo list for. Output: a todo list for that objective. Please be very clear what the objective is!",
        ),
    ]


def get_prompt(tools):
    prefix = """You are an AI who performs one task based on the following objective: {objective}. Take into account these previously completed tasks: {context}."""
    suffix = """Question: {task}
{agent_scratchpad}"""
    return ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["objective", "task", "context", "agent_scratchpad"],
    )
