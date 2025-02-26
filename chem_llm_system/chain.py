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

# LLM conf
client = OpenAI(api_key=key, base_url=base_url)

# global message history
message_history = []

def routing_function_orchestr(state):
    if state['done'] == 'validate':
        return 'validator'
    elif state["next_nodes"] == []:
        return 'orchestrator'
    elif state["next_nodes"][0][0] == 'automl':
        state['query'] = state["next_nodes"][0][1]
        return 'automl'
    elif state["next_nodes"][0][0] == 'chat_agent':
        state['query'] = state["next_nodes"][0][1]
        return 'chat_agent'

    
def routing_function_validator(state):
    if state['is_valid'] == True:
        return 'summarizer'
    else:
        return 'orchestrator'
    

def limit_message_history():
    """Ensure that message history only stores the last 2 requests."""
    if [i['role'] for i in message_history].count('user') > 2:
        message_history.pop(0)  # remove the oldest item
        while [i['role'] for i in message_history][0] != 'user':
            message_history.pop(0)

def decomposer(state):
    user_query = state["query"]
    
    prompt = f"""
    Break this query into several subtasks and return the JSON in the following format:
    {{"tasks": ["task 1", "task 2", "task 3"]}}.
    Your answer should only consist of a dictionary! Nothing extra.
    
    For example:
    Query: Predict the properties for the H2O molecule. Generate 5 molecules suitable for the treatment of Alzheimer's.
    Answer: {{"tasks": ["Predict the properties for the H2O molecule", "Generate 5 molecules suitable for the treatment of Alzheimer's"]}}
    Query: Train genrative models for molecules with hight Docking score.
    Answer: {{"tasks": ["Train genrative models for molecules with hight Docking score."]}}
    
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

    limit_message_history()  # Ensure only 2 queries are in history
    return {"tasks": subtasks, "pending_tasks": subtasks, "responses": []}

def automl_agent(state):
    query = state["query"]
    result = f"Automl not worked. Write to autor"
    message_history.append({"role": "assistant", "content": result})
    return state

def chat_agent_node(state):
    query = state["query"]
    result = f"Chat Agent not worked. Write to autor"
    message_history.append({"role": "assistant", "content": result})
    return state

def orchestrator(state):
    pending_tasks = state["pending_tasks"]
    responses = state["responses"]

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
    
    Your response must contain only json! No extra characters!
    """
    message_history.append({"role": "user", "content": task})
    messages = [{"role": "system", "content": tool_selection_prompt}] + message_history

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=messages,
        temperature=0.0,
        max_tokens=10000,
        extra_headers={"X-Title": "DrugDesign"},
    )

    # tool_data = json.loads(response.choices[0].message.content)
    tool_data = eval(response.choices[0].message.content)
    tools_to_use = tool_data.get("tools", [])
    message_history.append({"role": "assistant", "content": str(tool_data)})

    next_nodes = []
    for tool_info in tools_to_use:
        tool_name = tool_info.get("tool")
        tool_args = tool_info.get("arguments", {})

        if tool_name in ["gen_model", "pred_model", "database"]:
            # not agent, simple functions
            tool_funcs = {
                "gen_model": lambda query: f"Generated output: {query}",
                "pred_model": lambda query: f"Predicted output: {query}",
                "database": lambda query: f"Database result: {query}",
            }
            result = tool_funcs[tool_name](**tool_args)
            responses.append(result)
            message_history.append({"role": "assistant", "content": result})

        elif tool_name in ["automl", "chat_agent"]:
            # add info about launch next agent
            next_nodes.append((tool_name, {"query": tool_args["query"]}))
            
    if "next_nodes" not in state or state["next_nodes"] is None:
        state["next_nodes"] = []
            
    if not pending_tasks and not next_nodes:
        return {"done": "validate", "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses, "next_nodes": next_nodes}

def validator(state):
    responses = state["responses"]
    prompt = f"Validate the following responses:\n{responses}\nReturn JSON in the format: {{\"valid\": true or false, \"reason\": \"explanation\"}}"
    
    # response = client.chat.completions.create(
    #     model="meta-llama/llama-3.1-70b-instruct",
    #     messages=message_history + [{"role": "user", "content": prompt}],
    #     temperature=0.0,
    #     max_tokens=200,
    #     extra_headers={"X-Title": "DrugDesign"},
    # )
    
    # validation_result = json.loads(response.choices[0].message.content)
    # limit_message_history()  # Ensure only 2 queries are in history

    # if not validation_result.get("valid", False):
        # return {"done": False, "pending_tasks": state["pending_tasks"], "responses": []}
    state['is_valid'] = True
    return state

def summarizer(state):
    responses = state["responses"]
    prompt = f"Summarize the following information:\n{responses}"

    if len(responses) > 1:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            extra_headers={"X-Title": "DrugDesign"},
        ).choices[0].message.content
    else:
        response = responses[0]

    return {"summary": response}

# define agents graph
graph = Graph()

graph.add_node("decomposer", decomposer)
graph.add_node("orchestrator", orchestrator)
graph.add_node("automl", automl_agent)
graph.add_node("chat_agent", chat_agent_node)
graph.add_node("validator", validator)
graph.add_node("summarizer", summarizer)

graph.add_edge("decomposer", "orchestrator")
graph.add_conditional_edges(
    "orchestrator", routing_function_orchestr
)

graph.add_edge("automl", "orchestrator")
graph.add_edge("chat_agent", "orchestrator")

graph.add_conditional_edges(
    "validator",
    routing_function_validator
)

graph.add_edge("summarizer", END)
graph.set_entry_point("decomposer")

app = graph.compile()

if __name__ == "__main__":
    user_input = {"query": "Train genrative models for molecules with hight IC50"}
    result = app.invoke(user_input)
    print(result)
