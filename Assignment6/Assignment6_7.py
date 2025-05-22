
def DisplayMax(num):
    number = []
    print("Enter the numbers: ")
    for i in range(num):
        n = int(input())
        number.append(n)
        
    maximum = number[0]
    for i in range(1, len(number)):
        if number[i] > maximum:
            maximum = number[i]
    return maximum        
        
                               
          
def main():
    
    Value1 = 0
    
    print("Enter a number: ")
    Value1 = int(input())
    
    iRet = DisplayMax(Value1)
    print("The maximum number is: ", iRet)
   
     
if __name__ == "__main__":
    main()



