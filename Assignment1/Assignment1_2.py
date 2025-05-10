##########################################################
# File Name :   Assignment1_2.py
# Description : Used to check whether number is odd or even
# Author :      Nikita Shahaji Omase
# Date :        10/05/2025
###########################################################

def ChkNum(num):
    if((num % 2)==0):
        print("Even number")
    else:
        print("Odd number") 
           
def main():
    print("Enter number : ")
    value = int(input()) 
    
    ChkNum(value)  
     
if __name__ =="__main__":
    main()