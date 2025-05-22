def Factorial(num):
    fact = 1
    for i in range(1,num+1):
        fact = fact * i;
    return fact    
           
def main():
    Value = 0 
    print("Enter a number : ")
    Value = int(input())
    iRet = Factorial(Value)
    print("Factorial of number is :",iRet)
     
if __name__ == "__main__":
    main()
