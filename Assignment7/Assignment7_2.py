double = lambda x : x * 2
       
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
    
    iRet = list(map(double,li))
    print("Map output : ",iRet)   
           
if __name__ == "__main__":
    main()