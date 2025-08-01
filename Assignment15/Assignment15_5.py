import sys
import os

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <file_name> <search_string>")
        return

    file_name = sys.argv[1]
    search_str = sys.argv[2]

    if not os.path.exists(file_name):
        print(f"File '{file_name}' does not exist.")
        return

    try:
        with open(file_name, 'r') as f:
            content = f.read()
    except Exception as e:
        print("Error reading file:", e)
        return

    count = 0
    for i in range(len(content) - len(search_str) + 1):
        match = True
        for j in range(len(search_str)):
            if content[i + j] != search_str[j]:
                match = False
                break
        if match:
            count += 1

    if count == 0:
        print(f"'{search_str}' not found in the file.")
    else:
        print(f"'{search_str}' found {count} time(s) in the file.")

if __name__ == "__main__":
    main()
