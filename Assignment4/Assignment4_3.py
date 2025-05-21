
from functools import reduce

ChkNumber = lambda No: (No >= 70 and No <= 90)
Increase = lambda No : No + 10
Mult= lambda A , B : A * B

def main():
    No = 0
    num = 0
    print("Enter a number: ")
    No = int(input())
    li = []
    print("Enter number: ")
    for i in range(No):
       
        num = int(input())
        li.append(num)
    print("List of numbers: ",li)
    
    
  
    fData =list(filter(ChkNumber,li))   #typecasting
    print("Filter output : ",fData)

    mData = list(map(Increase,fData))
    print("Map output",mData)

    rData = reduce(Mult,mData)
    print("Reduce output : ",rData)
        
      
if __name__ =="__main__":
    main()    