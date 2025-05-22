Even = lambda x : ((x % 2)==0)
       
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
    
    iRet = list(filter(Even,li))
    print("filter output to keep even numbers : ",iRet)   
           
if __name__ == "__main__":
    main()