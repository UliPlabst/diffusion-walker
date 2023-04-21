import os

prompt_file = None
prompts_transformer = None

def set_prompts_transformer(t):
  global prompts_transformer
  prompts_transformer = t

def set_prompt_file(file):
  global prompt_file
  if(os.path.exists(file) == False):
    raise Exception("Prompt file does not exist")
  prompt_file = file

def get_prompts():
  if(prompt_file is None):
    raise Exception("Prompt file has not been set")
  with open(prompt_file, 'r') as f:
    lines = f.read().splitlines()  # split lines into an array
      
    lines = list(filter(lambda x: x and not x.isspace(), lines))  
    if(prompts_transformer is not None):
      lines = [ prompts_transformer(l) for l in lines ]
    print(f"Read {len(lines)} prompts")
    
    return lines
    # return non_empty_lines[:10]
    
prompts = None
prompt_index = 0

def set_prompt_index(idx):
  global prompt_index
  prompt_index = idx

def get_next_prompt():
   global prompt_index
   global prompts
   if(prompts is None):
    prompts = get_prompts()
   if(prompt_index == len(prompts)):
    return None
   res = prompts[prompt_index]
   print(f"@@ [prompt {prompt_index}]: {res}")
   prompt_index += 1
   return res