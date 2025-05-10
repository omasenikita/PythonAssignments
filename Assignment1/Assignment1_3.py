##########################################################
# File Name :   Assignment1_3.py
# Description : Used to print addition
# Author :      Nikita Shahaji Omase
# Date :        10/05/2025
###########################################################

def Add(n1,n2):
    ans = 0
    ans = n1 + n2
    
    return ans

def main():
    print("Enter first number:")
    num1 = int(input())
    
    print("Enter second number:")
    num2 = int(input())
    
    ret = Add(num1,num2)
    print("Addition is",ret)

if __name__ =="__main__":
    main()