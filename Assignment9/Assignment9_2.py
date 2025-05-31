import multiprocessing
import multiprocessing.process
import time

def Square(l5):
    for i in l5:
        print(i ** 2)  

def main():
    l1 = [2,4,5]
    l2 = [4, 6, 8, 7, 5, 4, 3]
    l3 = [7,43,23,54,3,265,43,3332]
    P1 = multiprocessing.Process(target=Square,args=(l1,))
    P2 = multiprocessing.Process(target=Square,args=(l2,))
    P3 = multiprocessing.Process(target=Square,args=(l3,))
    
    P1.start()
    P1.join()
    
    P2.start()
    P2.join()
    
    P3.start()
    P3.join()
    

if __name__ == "__main__":
    main()