import threading
import time
import multiprocessing

def Display():
    isum = 0
    for i in range(1000000):
        isum = isum + i
    print("Sum is: ", isum)

def main():
    
    start_time = time.time()
    T1 = threading.Thread(target=Display)
    end_time = time.time()
    print("Time taken by Thread: ", end_time - start_time)
    
    start_time = time.time()
    P1 = multiprocessing.Process(target=Display)
    end_time = time.time()
    print("Time taken by Process: ", end_time - start_time)
    start_time = time.time()
    
    Display()
    end_time = time.time()
    print("Time taken by Normal Function: ", end_time - start_time)
    T1.start()
    T1.join()
    P1.start()
    P1.join()
   
    


if __name__=="__main__":
    main()