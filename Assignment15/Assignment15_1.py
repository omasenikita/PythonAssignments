import os

def main():
  
    print("Enter the name of directory:")
    DirName = input()
    
    ret = os.path.abspath(DirName)
    if ret:
        print("The file is  present in current directory.")
        
    else:
         print("The file  not is present in current directory.")
    
    
if __name__ == "__main__":
    main()