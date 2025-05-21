
##########################################################
# File Name :   Assignment3_4.py
# Description : Used to find the Frequency of perticular element
# Author :      Nikita Shahaji Omase
# Date :        15/05/2025
###########################################################

def FindFrequency(num):
    li = []
    n1 = sum = src = 0
    
    print("Enter the element")
    for i in range (num):
        n1 = int(input())
        li.append(n1)
        
        print("Enter the number of search element")
        search = int(input())
        if(li[i] == search):
            sum = sum + 1      
        
        return sum
                
def main():
    
    
    no = 0
    iRet = 0
    print("Enter the number of elements")
    no = int(input())

    iRet = FindFrequency(no)
    print("Number present : ",iRet," times")
   
if __name__ == "__main__":
    main()