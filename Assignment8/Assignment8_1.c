
import threading

def Odd(): 
    num = 1
    iCnt = 1
    print("Odd")
    while(iCnt <= 10):
        
        if((num%2)!=0):
            
            print(num)
            iCnt= iCnt+1
        num = num + 1
   
def Even(): 
    num = 1
    iCnt = 1
    print("Even")
    while(iCnt <= 10):
        
        if((num%2)==0):
            print(num)
            iCnt= iCnt+1
        num = num + 1
  
def main():
    
    thread1 = threading.Thread(target=Odd)
    thread2 = threading.Thread(target = Even)
    
    thread1.start()
    thread1.join()
    thread2.start()
    
   
    thread2.join()
    
    
if __name__=="__main__":
    main()    
