from tkinter import Tk, filedialog
import os

def file_browser(filetypes:dict):
    '''
    return file path
    '''
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    filenames = filedialog.askopenfilenames(filetypes=filetypes['video files'])
    if len(filenames) > 0:
        root.destroy()
        result = list(filenames)
        # return str(filenames)
    else:
        filename = "Files not seleceted"
        root.destroy()
        result = str(filename)
    
    return result

def folder_browser():
    '''
    return folder
    '''
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()

    folder_path = filedialog.askdirectory()
    if folder_path:
        if os.path.isdir(folder_path):
            root.destroy()
            return str(folder_path)
        else:
            root.destroy()
            return str(folder_path)
    else:
        folder_path = "Folder not seleceted"
        root.destroy()
        return str(folder_path)
        
