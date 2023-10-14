
import pathlib
import string 
import json
import chardet
import PyPDF2


class FileLoader:
    """
    A class to:
        - Identify the file type of a file, 
        - Load it 
        - potentially perform conversion to ascii text
        - potentially perform some cleaning action
        - return it as a dictionary where you have the ascii text as well as the metadata about the file
    """
    def __init__(self):
        pass

    @staticmethod
    def is_file_contains_text(file_string:str) -> bool: # from https://stackoverflow.com/questions/1446549/how-to-identify-binary-and-text-files-using-python
        """ 
        Check if a file contains text or is binary
        Returns True if the file is text, False if it is binary or None if the file does not exist or is empty
        """
        try:
            s=open(file_string).read(1024)
        except Exception as e:
            print(f"WARNING Exception {e}: is_file_contains_text() could not open the file: {file_string} ")
            return None
        text_characters = "".join(map(chr, range(32, 127)) + list("\n\r\t\b"))
        _null_trans     = string.maketrans("", "")
        if not s:          # Empty files are considered as None
            return None
        if "\0" in s:      # Files with null bytes are likely binary
            return False
        # Get the non-text characters (maps a character to itself then
        # use the 'remove' option to get rid of the text characters.)
        t = s.translate(_null_trans, text_characters)
        # If more than 30% non-text characters, then
        # this is considered a binary file
        if float(len(t))/float(len(s)) > 0.30:
            return False
        return True
    

    @staticmethod
    def get_file_meta_data(file_string:str) -> dict:
        """ get the metadata of a file """
        file_path   = pathlib.Path(file_string)
        file_name   = file_path.name
        file_ext    = file_path.suffix
        file_length = file_path.stat().st_size
        return {'file_path':file_path ,
                'file_name': file_name,
                'file_ext':file_ext, 
                'file_length': file_length}


    @staticmethod
    def load_text_file(file_string:str, target_encoding:str=None ) -> dict:
        """ load a text file and return the text"""
        res  = FileLoader.get_file_meta_data(file_string)
        text = None
        # ensure content is not binary
        if FileLoader.is_file_contains_text(file_string): 
            with open(file_string, 'r') as f:      # we have text
                text = f.read()
            original_encoding = chardet.detect(text)['encoding']
            if target_encoding is not None and original_encoding != target_encoding:
                text = text.encode(original_encoding).decode(target_encoding, 'ignore') # convert to target_encoding
                encoding = target_encoding
            else:
                encoding = original_encoding
    
        res['original_encoding'] = original_encoding
        res['encoding']          = encoding
        res['text']              = text
        return res


    @staticmethod
    def load_json_file(file_string:str) -> dict:
        """ load a json file and return the text"""
        res          = FileLoader.get_file_meta_data(file_string)
        json_content = None
        try:
            with open(file_string) as f: json_content = json.load(f)
        except Exception as e:
            print(f"WARNING Exception {e}: load_json_file() could not open or load the JSON file: {file_string} ")
        
        res['json_content'] = json_content
        return res
        
    @staticmethod
    def load_pdf_file(file_string:str) -> dict:
        """ load a pdf file and return the text"""
        res  = FileLoader.get_file_meta_data(file_string)
        text = None
        res['pages'] = 0
        try:
            with open(file_string, 'rb') as f:      # we have text
                pdf_reader   = PyPDF2.PdfFileReader(f)
                res['pages'] = pdf_reader.numPages
                text         = ''
                for page_num in range(pdf_reader.numPages):
                    page  = pdf_reader.getPage(page_num)
                    text += page.extractText()
        except Exception as e:
            print(f"WARNING Exception {e}: load_pdf_file() could not open or load the PDF file: {file_string} ")
        
        res['text'] = text
        return res
    

    @staticmethod
    def load_file(file_string:str, target_encoding:str=None ) -> dict:
        """ identify file type load it and return the results as a dictionary with the file metadata and the"""

        if res['file_ext'].lower() == '.pdf':
            res = FileLoader.load_pdf_file(file_string)
        elif res['file_ext'].lower() == '.json':
            res = FileLoader.load_json_file(file_string)
        else:
            res = FileLoader.load_text_file(file_string, target_encoding)
        return res