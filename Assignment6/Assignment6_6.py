def Display(row):
    
    for i in range(1,row+1):
        for j in range(1,i+1):
            print("*",end= "")
        print()    
                              
          
def main():
    
    value1 = value2 = 0 
    
    print("Enter a row : ")
    Value1 = int(input())
    
    Display(Value1)
   
     
if __name__ == "__main__":
    main()


    
