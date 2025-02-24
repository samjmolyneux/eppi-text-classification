import os

def main():
    # Get the current directory
    current_dir = os.getcwd()

    # Iterate over all files in the directory
    for filename in os.listdir(current_dir):
        # Check if it's a Python file that starts with 'push' and does not end with 'to_azure.py'
        if filename.startswith("push") and filename.endswith(".py") and not filename.endswith("to_azure.py"):
            # Create the new filename by adding 'to_azure' before the '.py'
            new_filename = filename[:-3] + "_to_azure.py"
            # Rename the file
            os.rename(os.path.join(current_dir, filename), os.path.join(current_dir, new_filename))
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    main()
