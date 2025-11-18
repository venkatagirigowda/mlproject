import os
import pickle


def save_object(file_path,obj):
    dir_path=os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file:
        pickle.dump(obj,file)