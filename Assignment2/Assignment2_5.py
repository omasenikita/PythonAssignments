def Prime(num):
    for iCnt in range(2, num-1):
        
        if((num % iCnt )!= 0):
            return True;
        else:
            return False;
        
            
            
    return sum
               
def main():
    No1 = 0
    print("Enter first number: ")
    No1 = int(input())
    Ans = Prime(No1)
    if(Ans == True):
        print("Number is prime")
    else:
        print("Number is not prime")
    
        
if __name__ =="__main__":
    main()