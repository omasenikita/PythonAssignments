import os

def main():
  
    print("Enter the name of file u want to create:")
    fName = input()
    
    fname =open(fName, 'r')
    
    data = fname.read()
    if not data:
        print("File is empty")
    else:
        print("File is not empty")
        print("File content is:")
        print(data)
                 
    fname.close()  
    
if __name__ == "__main__":
    main()