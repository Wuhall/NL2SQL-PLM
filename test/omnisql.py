import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

input_prompt_template = '''Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{db_details}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query.'''

db_details = '''
Table: employees
- employee_id (INTEGER, PRIMARY KEY)
- first_name (TEXT)
- last_name (TEXT)
- department_id (INTEGER)
- hire_date (TEXT)
- salary (REAL)

Table: departments
- department_id (INTEGER, PRIMARY KEY)
- department_name (TEXT)
- manager_id (INTEGER, FOREIGN KEY REFERENCES employees(employee_id))

Table: projects
- project_id (INTEGER, PRIMARY KEY)
- project_name (TEXT)
- start_date (TEXT)
- end_date (TEXT)
- budget (REAL)

Table: employee_projects
- employee_id (INTEGER, FOREIGN KEY REFERENCES employees(employee_id))
- project_id (INTEGER, FOREIGN KEY REFERENCES projects(project_id))
- hours_worked (REAL)
- PRIMARY KEY (employee_id, project_id)
'''

question = '''
Find the names of all employees in the 'Marketing' department who have worked more than 100 hours on any project, 
along with the project names and total hours they worked on those projects. 
Order the results by total hours worked in descending order.
'''

def get_device():
    """获取可用的设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
device = get_device()

prompt = input_prompt_template.format(db_details = db_details, question = question)
model_path = "seeklhy/OmniSQL-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16
).to(device)

chat_prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt = True, tokenize = False
)

inputs = tokenizer([chat_prompt], return_tensors="pt")
inputs = inputs.to(model.device)

output_ids = model.generate(
    **inputs,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = 2048
)

input_len = len(inputs.input_ids[0])
output_ids = output_ids[0][input_len:]

response = tokenizer.batch_decode([output_ids], skip_special_tokens = True)[0]
print(response)
