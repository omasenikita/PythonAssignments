##########################################################
# File Name :   Assignment1_8.py
# Description : Used to print Star(*) pattern 
# Author :      Nikita Shahaji Omase
# Date :        10/05/2025
###########################################################

def Display(num):
    for i in range(num):
        print("* ",end=" ")
    
def main():
    print("Enter number : ")
    value = int(input())
    
    Display(value)  
     
if __name__ =="__main__":
    main()