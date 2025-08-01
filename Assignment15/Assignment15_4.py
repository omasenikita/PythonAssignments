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
    Content_1 = src.read()
    
    Content_2 = open(destination_file, 'r').read()
    
    if Content_1 == Content_2:
        print("Content both have same content.")
        
    else:
        print("Content both have different content.")
    
if __name__ == "__main__":
    main()
