from functools import reduce

Product = lambda A, B : A * B
       
def main():
    num = 0
    
    print("Enter a number")
    num = int(input())
    
    li = []
    print("Enter a number")
    for i in range(1,num+1):
        n= int(input())
        li.append(n)
        
    print("List ",li) 
    
    iRet = int(reduce(Product,li))
    print("reduce output to calculate product of all elements : ",iRet)   
           
if __name__ == "__main__":
    main()