from MarvellousNum import ChkPrime 

def ListPrime(num):
    prime = []
    for i in range(2, num + 1):
        if ChkPrime(i):
            prime.append(i)
    return prime

def main():
   num = 0
   print("Enter a number: ")
   num = int(input())
   prime = ListPrime(num)
   print("List of prime numbers: ", prime)  
    
if __name__ == "__main__":
    main()    