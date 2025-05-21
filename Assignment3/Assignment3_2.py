
##########################################################
# File Name :   Assignment3_2.py
# Description : Used to calculate maximum element for list 
# Author :      Nikita Shahaji Omase
# Date :        15/05/2025
###########################################################

def FindMax(num):
    li = []
    n1 = sum = 0
    
    print("Enter the element")
    for i in range (num):
        n1 = int(input())
        li.append(n1)
        max = li[0]
        if (li[i] > max):
            
            max = li[i]
    return max
            
def main():
    
    no = 0
    iRet = 0
    print("Enter the number of elements");
    no = int(input())
    iRet = FindMax(no)
    print("maximum of elements are : ",iRet);
   
if __name__ == "__main__":
    main()