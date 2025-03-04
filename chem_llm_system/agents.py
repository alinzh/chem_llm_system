from dataset_handler.chembl.chembl_utils import ChemblLoader
import yaml
from openai import OpenAI
import json


with open("./chem_llm_system/conf/conf.yaml", "r") as file:
    conf = yaml.safe_load(file)
    key = conf["api_key"]
    base_url = conf["base_url"]
    file_path = './chem_llm_system/dataset_handler/chembl/p1_short.csv'


def orchestrator(state):
    pending_tasks = state["pending_tasks"]
    responses = state["responses"]

    if not pending_tasks:
        return {"done": "validate", "responses": responses}

    user_query = pending_tasks.pop(0)

    chembl_client = ChemblLoader(True, file_path)

    prompt = r"""
      Extract relevant dataset filtering parameters from the following user request.
        Available columns:""" +  str(chembl_client.get_columns()) + r"""
        Return JSON in the format: {"selected_columns": [...], "filters": {{...}}}
        It is not necessary to specify filters (the dictionary with filters can be empty. In filters, you can specify ranges, string values, and Booleans.
        Required: Your answer must contain only language vocabulary (start answer from "{"). Use "float('-inf')" and "float('inf')" for negative and positive infinity (not None!).
        
        For example:
        User request: "Show molecules with molecular weight between 150 and 500."
        You: {'selected_columns': ['Molecular Weight"], "filters": {"Molecular Weight": (150, 500)}}
        Or:
        User request: "Show small molecules."
        You: {"selected_columns": ["Molecular Weight", "Type"], "filters": {"Type": "Small molecule"}}
        
        User request: 
        """ + user_query + r"""You: """
        
    llm_client = OpenAI(api_key=key, base_url=base_url)
        
    response = llm_client.chat.completions.create(
        model="meta-llama/llama-3.1-70b-instruct",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
    )
    print('========')    
    print(response.choices[0].message.content)  
    print('========')   
        
    try:
        query_params = eval(response.choices[0].message.content)
    except:
        try:
            query_params = eval(response.choices[0].message.content.split('You: ')[-1])
        except:
            return {"done": "error", "message": "Failed to parse LLM response.", "responses": None}
        
    selected_columns = query_params.get("selected_columns", [])
    filters = query_params.get("filters", {}) 
  
    result_df = chembl_client.get_filtered_data(selected_columns, filters)
    responses.append(result_df)
        
    if not pending_tasks:
        return {"done": "validate", "responses": responses}
    else:
        return {"done": False, "pending_tasks": pending_tasks, "responses": responses}
        
if __name__ == "__main__":
    querys = [
        """Find molecules that contain all following properties:
        ChEMBL ID
        Name
        Molecular Weight (between 250 and 500 Da)
        AlogP (between -2 and 5)
        Polar Surface Area (PSA) (between 20 and 150 Å²)
        #RO5 Violations (exactly 0 or 1)
        CX LogP (between -1 and 6)
        Aromatic Rings (between 0 and 6)
        Heavy Atoms (between 15 and 20)
        Molecular Formula
        """, 
        "Connections with a number of rotatable bonds of no more than 5 and a positive LogD are required.", 
        "Find molecules with positive LogD.", 
        "Molecules with a polar surface area (PSA) of less than 100 are needed"
    ]      
    
    for q in querys:
        print('========')
        print('Query:', q)
        res = orchestrator({"pending_tasks": [q], "responses": []})
        print(res["responses"][0])
        print(res["responses"][0].iloc[0])