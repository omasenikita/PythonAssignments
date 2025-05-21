
def Display(Num):
    for i in range(Num):  
        for j in range(Num):
            print("* ",end = " ")    
            
        print()   

def main():
    No1 = 0
    print("Enter first number: ")
    No1 = int(input())
    Display(No1)
     
    
if __name__ =="__main__":
    main()