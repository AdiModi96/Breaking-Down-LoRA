import os
import inspect

current_file_path = os.path.realpath(inspect.getfile(inspect.currentframe()))
project_folder_path = os.path.dirname(current_file_path)

datasets_folder_path = os.path.join(project_folder_path, 'datasets')
if not os.path.isdir(datasets_folder_path):
    os.makedirs(datasets_folder_path)

models_folder_path = os.path.join(project_folder_path, 'models')
if not os.path.isdir(models_folder_path):
    os.makedirs(models_folder_path)
