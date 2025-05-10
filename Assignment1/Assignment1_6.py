#########################################################
# File Name :     Assignment1_6.py
# Description :   Used to print Postive ,negative or zero
# import file:    Numchk.py 
# import function:NumCh 
# Author :        Nikita Shahaji Omase
# Date :          10/05/2025
###########################################################

from Numchk import NumCh
def main():
    print("Enter number : ")
    value = int(input()) 
    
    NumCh(value)  
     
if __name__ =="__main__":
    main()