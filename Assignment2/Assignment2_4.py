def FactorAddition(n):
    sum = 0
    for i in range(1, n ):
        if(n % i== 0):
            sum = sum + i
            
            
    return sum
               
def main():
    No1 = 0
    print("Enter first number: ")
    No1 = int(input())
    Ans = FactorAddition(No1)
    print("sum of factor of number is",Ans)
        
if __name__ =="__main__":
    main()