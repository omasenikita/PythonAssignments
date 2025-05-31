import threading
import time

def Display():
    for i in range(1,5):
        time.sleep(1)
        print(i)   

def main():
    thread1 = threading.Thread(target=Display)
    thread2 = threading.Thread(target=Display)
    thread3 = threading.Thread(target=Display)
    
    thread1.start()
    thread1.join()
    
    thread2.start()
    thread2.join()
    
    thread3.start()
    thread3.join()
    
    

if __name__ == "__main__":
    main()
