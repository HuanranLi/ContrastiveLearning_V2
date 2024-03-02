import os
import glob
import shutil

def rename_files_in_metrics(directory_path):
    for root, dirs, files in os.walk(directory_path):
        # Check if the current folder is named 'metrics'
        if os.path.basename(root) == 'metrics':
            # Search for files starting with 'test' and ending with 'accuracy'
            for filename in glob.glob(os.path.join(root, 'test*accuracy')):
                # Create the new filename
                new_filename = filename + '_100'
                # Copy the file
                # shutil.copyfile(filename, new_filename)
                print(f'Copy {filename} \nto\n {new_filename}\n under path\n {root}\n\n')

# Example usage
directory_path = '../logs/'  # Replace with your directory path
rename_files_in_metrics(directory_path)
