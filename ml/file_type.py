import os
from collections import Counter

def count_file_types(folder_path):
    """
    Count the occurrences of different file extensions in a folder and its subdirectories.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    Counter: A Counter object containing counts of file extensions.
    """
    file_extension_counter = Counter()

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, extension = os.path.splitext(file)
            file_extension_counter[extension] += 1

    return file_extension_counter

def delete_non_jpg_files(folder_path):
    """
    Delete files with extensions other than '.jpg' in a folder and its subdirectories.

    Parameters:
    - folder_path (str): The path to the folder containing files.

    Returns:
    None
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension.lower() != ".jpg":
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    """
    Main script for counting and managing file types in folders.

    This script performs the following steps:
    1. Iterates through clothing classes to process their folders.
    2. Counts and displays file type occurrences.
    3. Offers an option to delete non-JPG files.
    """

    # List of clothing classes to process
    classes = ['hoodies', 'hoodies-female', 'longsleeve', 'shirt', 'sweatshirt', 'sweatshirt-female']

    # Iterate through each clothing class to process folders
    for cloth_class in classes:
        folder_path = f'zalando/{cloth_class}'
        if os.path.isdir(folder_path):
            # Count file type occurrences and display results
            file_extension_counter = count_file_types(folder_path)
            print(f"File Type Counts for {cloth_class}:")
            for extension, count in file_extension_counter.items():
                print(f"{extension}: {count}")

            # Ask user if they want to delete non-JPG files
            if input("Do you want to delete non-JPG files? (yes/no): ").lower() == "yes":
                delete_non_jpg_files(folder_path)
                print("Deleted.")
        else:
            print("Invalid folder path.")
