
def CountNumber(Num):
    Digit = 0
    sum = 0
    
    if(Num <= 0):
        Num = -Num
        
    if(Num == 0):
        return 1
             
    while Num > 0:
        Digit = Num % 10
        
        Num = Num // 10
        sum = sum + Digit
    return sum

def main():
    No = 0
    print("Enter first number: ")
    No = int(input())
    Ans = CountNumber(No)
    print("Number of digits in number is",Ans)     
    
if __name__ =="__main__":
    main()