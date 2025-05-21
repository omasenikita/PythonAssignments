from functools import reduce

def ChkPrime(n):
    if n <= 1:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True

def Multi(n):
    return n * 2

def Max(a, b):
    if a > b:
        return a
    else:
        return b


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
  
    fData =list(filter(ChkPrime,li))   #typecasting
    print("Filter output : ",fData)

    mData = list(map(Multi,fData))
    print("Map output",mData)

    rData = reduce(Max,mData)
    print("Reduce output : ",rData)
        
      
if __name__ =="__main__":
    main()    