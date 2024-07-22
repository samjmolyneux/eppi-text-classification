def check_module_import(module_name):
    try:
        __import__(module_name)
        print(f"Module '{module_name}' has been successfully imported.")
    except ImportError:
        print(f"Error: Module '{module_name}' could not be imported.")


# Example usage
if __name__ == "__main__":
    module_to_check = (
        "eppi_text_classification"  # Replace with the module you want to check
    )
    check_module_import(module_to_check)
