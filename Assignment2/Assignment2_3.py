def factorial(n):
    fact = 1
    for i in range(1, n+1):
        fact = fact * i
    return fact
               
def main():
    No1 = 0
    print("Enter first number: ")
    No1 = int(input())
    Ans = factorial(No1)
    print("factorial of number is",Ans)
        
if __name__ =="__main__":
    main()