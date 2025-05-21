
##########################################################
# File Name :   Assignment3_3.py
# Description : Used to calculate minimum element for list 
# Author :      Nikita Shahaji Omase
# Date :        15/05/2025
###########################################################

def FindMin(num):
    li = []
    n1 = sum = 0
    
    print("Enter the element")
    for i in range (num):
        n1 = int(input())
        li.append(n1)
        min = li[0]
        if (li[i] < min):
            
            min = li[i]
    return min
            
def main():
    
    no = 0
    iRet = 0
    print("Enter the number of elements");
    no = int(input())
    iRet = FindMin(no)
    print("minimum of elements are : ",iRet);
   
if __name__ == "__main__":
    main()