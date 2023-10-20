import json
import chardet

from   bs4 import BeautifulSoup as bs


class TextUtils():
  """
  TextUtils is a class that regroup the methods to clean an manipulate text mainly for
  inputs and outputs of models and allow to perform. 
     - Charset conversion
     - cleaning of strings for commandline
     - extraction of json from text
     - extraction of text from html

  """

  def __init__(self) -> None:
    
    pass


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
    


   
    