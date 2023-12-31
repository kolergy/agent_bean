import json
import chardet

from typing import Dict
from   bs4  import BeautifulSoup as bs


class TextUtils():
  """
  TextUtils is a class that regroup the methods to clean an manipulate text mainly for
  inputs and outputs of models and allow to perform. 
     - Charset conversion
     - cleaning of strings for commandline
     - extraction of json from text
     - extraction of text from html
  """

  def __init__(self, action:str, text_input:str) -> None:
    self.action     = action
    self.text_input = text_input

  
"""
  def NOT_READY__init__(self, prompt_recepee:Dict, prompt_inputs:[str]) -> None:
    self.prompt_template  = None
    self.prompt_inputs    = prompt_inputs
    self.assembled_prompt = None
    self.prompt_recepee   = prompt_recepee
    self.prompt_target    = None  # LLM, comandline, files, etc
    self.prompt_len_char  = 0
    self.prompt_len_tok   = 0
    self.prompt_sucess    = 0
    
"""

  def NOT_READY_apply_recepee(self) -> None:
    """
    apply the prompt recepee to fetch the template and assemble the prompt with the inputs
    """
    self.prompt_template      = self.prompt_recepee['template']
    self.prompt_target        = self.prompt_recepee['target_use']
    self.prompt_input_mapping = self.prompt_recepee['input_mapping']
    if len(self.prompt_inputs) < len(self.prompt_input_mapping):
      raise ValueError("Not enough inputs for the input mapping")
    #self.assembled_prompt     = ''.join(action_params['prompt_system']).format(code_language=code_language) 
            
    #self.prompt_len_char = 
    #self.prompt_len_tok  = 
    


  @staticmethod
  def extract_text_from_html(self, string:str) -> str:
    """
    Extract the usefull text from an html string using BS4
    """
    soup = bs(string, features="html.parser")
    return soup.get_text()
    

  @staticmethod
  def convert_charset_to_ascii(self, string:str) -> str:
    """
    identify charset and Convert to ASCII
    """
    charset = chardet.detect(string)
    return string.decode(charset['encoding']).encode('ascii', 'ignore').decode('ascii')


  @staticmethod
  def clean_string_for_commandline(self, string) -> str:
    """
    Clean a string for commandline
    """
    string_ascii = TextUtils.convert_charset_to_ascii(string)
    return string_ascii.replace("'", "\\'")

  @staticmethod
  def clean_string_for_filename(self, string) -> str:
    """
    Clean a string for a valid filename
    """
    string_ascii = TextUtils.convert_charset_to_ascii(string)
    string_ascii.replace(" " , "_")
    string_ascii.replace("'" , "_")
    string_ascii.replace('"' , "_")
    string_ascii.replace("/" , "_")
    string_ascii.replace("\\", "_")
    string_ascii.replace(":" , "_")
    string_ascii.replace("*" , "_")
    string_ascii.replace("?" , "_")
    string_ascii.replace("<" , "_")
    string_ascii.replace(">" , "_")
    string_ascii.replace("|" , "_")
    return string_ascii

  @staticmethod
  def extract_json_from_text(self, string:str) -> dict:
    """
    identify the JSON among the text then Extract it and return a dictionary from the json
    the Json are delimited by the following set of characters: #0-=#
    """
    json_string = string.split("#0-=#")[1]
    return json.loads(json_string)
    


   
    