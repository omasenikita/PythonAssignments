##########################################################
# File Name :   Assignment1_7.py
# Description : Used to check number is divisible by 5 or not
# Author :      Nikita Shahaji Omase
# Date :        10/05/2025
###########################################################

def Divisible(num):
    if((num % 5)==0):
        print("true")
    else:
        print("false")    
def main():
    print("Enter number : ")
    value = int(input()) 
    
    Divisible(value)  
     
if __name__ =="__main__":
    main()