import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <source_file> <destination_file>")
        
        print("Copy one file's content to another without using any special library function.")
        return

    source_file = sys.argv[1]
    destination_file = sys.argv[2]

    if not os.path.exists(source_file):
        print(f"Source file '{source_file}' does not exist.")
        return

    if not os.path.isfile(source_file):
        print(f"Source file '{source_file}' is not a valid file.")
        return

    
    src = open(source_file, 'r')
    content = src.read()

    # Open destination file in write mode to create or overwrite
    dest= open(destination_file, 'w')
    dest.write(content)

    print(f"Content copied from '{source_file}' to '{destination_file}' successfully.")

   
if __name__ == "__main__":
    main()
