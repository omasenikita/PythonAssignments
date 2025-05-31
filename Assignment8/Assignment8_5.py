
import threading

def Display():
    print("---------1 to 50---------------")
    for i in range(1,50+1):
        print(i)
    
    
def DisplayReverse():
    print("---------50 to 1-----------")
    for i in range(50,0,-1):
        print(i)
        

def main():
    
    thread1 = threading.Thread(target = Display) 
    thread2 = threading.Thread(target = DisplayReverse) 

    thread1.start()
    thread1.join()

    thread2.start()
    
    thread2.join()
    
if __name__ == "__main__":
    main()    