from Arithmetic import Add
from Arithmetic import Sub
from Arithmetic import Multi
from Arithmetic import Div


def main():
    Ans = 0
    No1 = 0
    No2 = 0
    print("Enter first number: ")
    No1 = int(input())
    print("Enter Second  number: ")
    No2 = int(input())
    Ans = Add(No1, No2)
    print("Addition of two numbers is: ",Ans)
    Ans = Multi(No1, No2)
    print("Multiplication of two numbers is: ",Ans)
    Ans = Sub(No1, No2)
    print("Subtraction of two numbers is: ",Ans)
    Ans = Div(No1, No2)
    print("Division of two numbers is: ",Ans)   
    
if __name__ =="__main__":
    main()
