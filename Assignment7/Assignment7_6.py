
def ChkPrime(num):
   
    prime = [] 
    icnt = 0
    is_prime = True    #Assume number is prime
    for i in range (2,int(num/2)+1):
        if((num % i)==0):
            is_prime = False
            break
    if(is_prime):
        prime.append(num)
    return prime    

def main():
    num = 0
    
    print("Enter a number")
    num = int(input())
    li = []
    print("Enter a number")
    for i in range(1,num+1):
        n= int(input())
        li.append(n)
    
    iRet = list(filter(ChkPrime,li))
    print("filter output to get prime numbers : ",iRet)   
           
if __name__ == "__main__":
    main()