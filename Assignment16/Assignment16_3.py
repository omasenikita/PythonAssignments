import os
def main():
    
    print("Enter the name of file that you want to check:")
    FileName = input()
    
    fobj = open(FileName,'r')
    data = fobj.read()
    
    count = 0
    print(data)
    for char in data:
        count  += 1
        
    print("Number of characters in the file is:", count)
    
    
    cnt = 0
    for line in range(len(data)):
        if data[line] == '\n':
            cnt += 1
            
    print("Number of lines in the file is:", cnt)
    
    cnt = 0
    for word in data.split():
        cnt += 1
        
    print("Number of words in the file is:", cnt)
    
        

       
      
     
if __name__ =="__main__":
    main()