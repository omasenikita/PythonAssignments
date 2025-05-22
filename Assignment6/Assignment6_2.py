def SumEven():
    num = 1
    sum = 0
    while(num <= 100):
        
        if((num % 2)==0):     
            sum = sum + num
        num = num + 1
    return sum               
        
def main():
   iRet =  SumEven()
   print("Sum of  Even numbers between 1 to 100 is : ",iRet)
     
if __name__ == "__main__":
    main()
