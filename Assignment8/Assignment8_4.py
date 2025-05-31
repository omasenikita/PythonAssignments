
import threading
def Small(ch):
    print("Thread name is :", threading.current_thread().name)
    print("Thread ID is :", threading.get_ident())
    icnt = 0
    for str1 in ch:
       if(str1 >= 'A' and str1 <='Z'):
           icnt = icnt+1
    print("Number of Capital letters are : ",icnt)
       
def Capital(ch):
    print("Thread name is :", threading.current_thread().name)
    print("Thread ID is :", threading.get_ident())
    icnt2 = 0
    for str1 in ch:
       if(str1 >= 'a' and str1 <='z'):
           icnt2 = icnt2+1
    print("Number of small letters are :",icnt2)
       
def Digit(ch):
    print("Thread name is :", threading.current_thread().name)
    print("Thread ID is :", threading.get_ident())
    icnt3 = 0
    for str1 in ch:
       if(str1 >= '0' and str1 <='9'):
           icnt3 = icnt3+1
    print("Number of Digits are :",icnt3)
        

def main():

    print("Enter a string")
    str1 = input()
    
    print("Thread name is :", threading.current_thread().name)
    print("Thread ID is :", threading.get_ident())
 
    T1 = threading.Thread(target = Small, args = (str1,))    
    T2 = threading.Thread(target = Capital, args = (str1,))    
    T3 = threading.Thread(target = Digit, args = (str1,))    

    T1.start()
    T2.start()
    T3.start()
    
    T1.join()
    T2.join()
    T3.join()
    print("End of main thread")
    
if __name__ == "__main__":
    main()    