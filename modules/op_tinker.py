from tkinter import Tk, filedialog
import os

def file_browser(data_type):
    '''
    data_type as Files or Folder
    '''
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    if data_type == "Files":
        filenames = filedialog.askopenfilenames()
        if len(filenames) > 0:
            root.destroy()
            return str(filenames)
        else:
            filename = "Files not seleceted"
            root.destroy()
            return str(filename)

    elif data_type == "Folder":
        filename = filedialog.askdirectory()
        if filename:
            if os.path.isdir(filename):
                root.destroy()
                return str(filename)
            else:
                root.destroy()
                return str(filename)
        else:
            filename = "Folder not seleceted"
            root.destroy()
            return str(filename)