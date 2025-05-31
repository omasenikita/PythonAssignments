
import threading

def OddList(iNo):
    iSum = 0
    for i in iNo:
        if(i%2 != 0):
            iSum = iSum + i
    print("Odd list sum",iSum)          

def EvenList(iNo):  
    iSum = 0
    for i in iNo:
        if(i%2 == 0):
            iSum = iSum + i
    print("Even list sum",iSum)  
            
  
def main():
    num = 0
    print("Enter a number : ")
    num = int(input())
    l1 = []
    for i in range(1, num + 1):
        no = int(input())
        l1.append(no)
    thread1 = threading.Thread(target = OddList, args = (l1,))
    thread2 = threading.Thread(target = EvenList,args = (l1,))
   
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    print("Exit From Main")   
    
if __name__=="__main__":
    main()    