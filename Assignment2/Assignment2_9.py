def CountNumber(Num):
    Digit = 0
    
    if(Num < 0):
        Num = -Num
    
    if (Num == 0):
        return 1
        
    while Num > 0:
        Num = Num // 10
        Digit = Digit +  1
    return Digit
           
def main():
    No = 0
    print("Enter first number: ")
    No = int(input())
    Ans = CountNumber(No)
    print("Number of digits in number is",Ans)     
    
if __name__=="__main__":
    main()