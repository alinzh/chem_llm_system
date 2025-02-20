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

# Global message history
message_history = []

def limit_message_history():
    """Ensure that message history only stores the last 2 requests."""
    if [i['role'] for i in message_history].count('user') > 2:
        message_history.pop(0)  # Remove the oldest item
        while [i['role'] for i in message_history][0] != 'user':
            message_history.pop(0)

def decomposer(inputs):
    user_query = inputs["query"]
    message_history.append({"role": "user", "content": user_query})
    limit_message_history()  # Ensure only 2 queries are in history
    
    prompt = f"""
    Break this query into several subtasks and return the JSON in the following format:
    {{"tasks": ["task 1", "task 2", "task 3"]}}.
    Your answer should only consist of a dictionary! Nothing extra.
    
    For example:
    Query: Predict the properties for the H2O molecule. Generate 5 molecules suitable for the treatment of Alzheimer's.
    Answer: {{"tasks": ["Predict the properties for the H2O molecule", "Generate 5 molecules suitable for the treatment of Alzheimer's"]}}
    
    Query: {user_query}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=2000,
        extra_headers={"X-Title": "DrugDesign"},
    )

    try:
        structured_output = json.loads(response.choices[0].message.content)
        subtasks = structured_output.get("tasks", [])
    except json.JSONDecodeError:
        text = response.choices[0].message.content
        match = re.search(r"(.*?)", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            structured_output = json.loads(json_str)
            subtasks = structured_output.get("tasks", [])
        else:
            subtasks = []

    message_history.append({"role": "assistant", "content": f"Decomposed tasks: {subtasks}"})
    limit_message_history()  # Ensure only 2 queries are in history
    return {"tasks": subtasks, "pending_tasks": subtasks, "responses": []}

def orchestrator(inputs):
    pending_tasks = inputs["pending_tasks"]
    responses = inputs["responses"]

    if not pending_tasks:
        return {"done": "validate", "responses": responses}

    task = pending_tasks.pop(0)
    
    tool_selection_prompt = f"""
    Determine which tools to use to complete the following task.
    Return the JSON in the following format: {{"tools": [{{"tool": "tool_name", "arguments": {{"query": "value"}}}}]}}.
    Available tools:
    - "gen_model": runs a generative model inference.
    - "pred_model": runs a predictive model inference.
    - "database": queries the database.
    - "automl": calls the AutoML agent.
    - "chat_agent": queries the chat agent.
    
    Task: {task}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=message_history + [{"role": "user", "content": tool_selection_prompt}],
        temperature=0.0,
        max_tokens=200,
        extra_headers={"X-Title": "DrugDesign"},
    )

    tool_data = json.loads(response.choices[0].message.content)
    tools_to_use = tool_data.get("tools", [])

    tools = {
        "gen_model": Tool(name="Generative Model", func=lambda query: f"Generated output: {query}", description="Runs a generative model inference."),
        "pred_model": Tool(name="Predictive Model", func=lambda query: f"Predicted output: {query}", description="Runs a predictive model inference."),
        "database": Tool(name="Database", func=lambda query: f"Database result: {query}", description="Queries the database."),
        "automl": Tool(name="AutoML", func=lambda query: f"AutoML output: {query}", description="Calls the AutoML agent."),
        "chat_agent": Tool(name="Chat Agent", func=lambda query: f"Chat response: {query}", description="Queries the chat agent."),
    }

    for tool_info in tools_to_use:
        tool_name = tool_info.get("tool")
        tool_args = tool_info.get("arguments", {})
        if tool_name in tools:
            result = tools[tool_name].func(**tool_args)
        else:
            result = f"Could not determine the tool for task: {task}"
        responses.append(result)
        message_history.append({"role": "assistant", "content": f"Tool {tool_name} executed with result: {result}"})
        limit_message_history()  # Ensure only 2 queries are in history

    if not pending_tasks:
        return {"done": "validate", "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses}

def validator(inputs):
    responses = inputs["responses"]
    prompt = f"Validate the following responses:\n{responses}\nReturn JSON in the format: {{\"valid\": true or false, \"reason\": \"explanation\"}}"
    
    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=message_history + [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
        extra_headers={"X-Title": "DrugDesign"},
    )
    
    validation_result = json.loads(response.choices[0].message.content)
    limit_message_history()  # Ensure only 2 queries are in history
    return {"validated": validation_result, "responses": responses}

def summarizer(inputs):
    responses = inputs["responses"]
    prompt = f"Summarize the following information:\n{responses}"

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=message_history + [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        extra_headers={"X-Title": "DrugDesign"},
    )

    limit_message_history()  # Ensure only 2 queries are in history
    return {"summary": response.choices[0].message.content}

# == GRAPH CREATION ==
graph = Graph()

graph.add_node("decomposer", decomposer)
graph.add_node("orchestrator", orchestrator)
graph.add_node("validator", validator)
graph.add_node("summarizer", summarizer)

graph.add_edge("decomposer", "orchestrator")
graph.add_conditional_edges("orchestrator", lambda x: [("validator")] if x.get("done") == "validate" else [("orchestrator")])
graph.add_edge("validator", "summarizer")
graph.add_edge("summarizer", END)
graph.set_entry_point("decomposer")

app = graph.compile()

if __name__ == "__main__":
    user_input = {"query": "Find new drug candidates and predict their efficacy"}
    result = app.invoke(user_input)
    print(result)

    # Asking an additional question after the initial flow
    additional_input = {"query": "Can you explain the predicted efficacy of these candidates?"}
    additional_result = app.invoke(additional_input)
    print(additional_result)
    
    additional_input = {"query": "What can you do?"}
    additional_result = app.invoke(additional_input)
    print(additional_result)
