
import threading

def OddFactors(iNo): 
    iOSum = 0 
    for i in range(2,iNo+1):
        if(i%2 != 0):

          if((iNo % i) == 0):
             iOSum = iOSum + i 
            
    print("Sum of odd factors",iOSum)  

def EvenFactors(iNo): 
    iESum = 0
    for i in range(2,iNo+1):
        if(i%2 == 0):
           if((iNo % i) == 0):
             iESum = iESum + i
             
    print("Sum of Even factors",iESum)          
            
  
def main():
    num = 0
    print("Enter a number : ")
    num = int(input())
    thread1 = threading.Thread(target = OddFactors, args = (num,))
    thread2 = threading.Thread(target = EvenFactors,args = (num,))
   
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    print("Exit From Main")   
    
if __name__=="__main__":
    main()    