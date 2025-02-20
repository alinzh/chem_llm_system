from openai import OpenAI
from langchain.tools import Tool
from langgraph.graph import Graph, END
import json
import re
import yaml

with open("./chem_llm_system/conf/conf.yaml", "r") as file:
    conf = yaml.safe_load(file)
    key = conf["api_key"]
    base_url = conf["base_url"]

# LLM Configuration
client = OpenAI(api_key=key, base_url=base_url)


def decomposer(inputs):
    user_query = inputs["query"]
    prompt = f"""
    Break this query into several subtasks and return the JSON in the following format:
    {{"tasks": ["task 1", "task 2", "task 3"]}}
    
    Query: {user_query}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        extra_headers={"X-Title": "DrugDesign"},
    )

    # Parse the JSON response from LLM
    try:
        structured_output = json.loads(response.choices[0].message.content)
        subtasks = structured_output.get("tasks", [])
    except json.JSONDecodeError:
        text = response.choices[0].message.content
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            structured_output = json.loads(json_str)  # Преобразуем в словарь
            subtasks = structured_output.get("tasks", [])
        else:
            subtasks = []

    return {"tasks": subtasks, "pending_tasks": subtasks, "responses": []}


def orchestrator(inputs):
    pending_tasks = inputs["pending_tasks"]
    responses = inputs["responses"]

    if not pending_tasks:
        # If there are no tasks, finish execution
        return {"done": True, "responses": responses}

    task = pending_tasks.pop(0)  # Take the next task

    # Determine which tool to use for the task
    tool_selection_prompt = f"""
    Determine which tool to use to complete the following task.
    Return the JSON in the following format: {{"tool": "tool_name", "arguments": {{"query": "value"}}}}.
    Available tools:
    - "search": searches for information based on the query.
    - "calc": performs mathematical calculations.

    Task: {task}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": tool_selection_prompt}],
        temperature=0.0,
        max_tokens=200,
        extra_headers={"X-Title": "DrugDesign"},
    )

    tool_data = json.loads(response.choices[0].message.content)

    tool_name = tool_data.get("tool")
    tool_args = tool_data.get("arguments", {})

    # Define available tools
    tools = {
        "search": Tool(
            name="Search",
            func=lambda query: f"Search result: {query}",
            description="Searches for information based on the query.",
        ),
        "calc": Tool(
            name="Calculator",
            func=lambda query: f"Calculation result: {query}",
            description="Performs mathematical calculations.",
        ),
    }

    # Execute the selected tool
    if tool_name in tools:
        result = tools[tool_name].func(**tool_args)
    else:
        result = f"Could not determine the tool for task: {task}"

    responses.append(result)

    # If all tasks are completed, return "done"
    if not pending_tasks:
        return {"done": True, "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses}


# == 3. SUMMARIZER: Summarizes the results ==
def summarizer(inputs):
    responses = inputs["responses"]
    prompt = f"Summarize the following information:\n{responses}"

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        extra_headers={"X-Title": "DrugDesign"},
    )

    return {"summary": response.choices[0].message.content}


# == GRAPH CREATION ==
graph = Graph()

# Add nodes
graph.add_node("decomposer", decomposer)
graph.add_node("orchestrator", orchestrator)
graph.add_node("summarizer", summarizer)

# Define connections
graph.add_edge("decomposer", "orchestrator")  # Decomposed tasks → Orchestrator

# Change the condition
graph.add_conditional_edges(
    "orchestrator", lambda x: [("summarizer")] if x.get("done") else [("orchestrator")]
)

graph.add_edge("summarizer", END)  # Final node
graph.set_entry_point("decomposer")  # Starting node

# Compile the graph
app = graph.compile()

# == EXECUTION ==
user_input = {"query": "Find the weather in Moscow and calculate 2+2"}
result = app.invoke(user_input)

print(result)
