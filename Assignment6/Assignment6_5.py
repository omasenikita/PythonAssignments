def ChkPrime(num):
    is_prime = False
    for i in range(2,num+1):
        if((num % i)==0):
            return True
        else:
            return False
          
def main():
    Value = 0 
    print("Enter a number : ")
    Value = int(input())
    iRet = ChkPrime(Value)
    if(iRet == True):
        print("number not a prime number")
    else:
        print("number is prime")
     
if __name__ == "__main__":
    main()
