class Arithmetic:
    
    def __init__(self, A):
        self.value = A

    def ChkPrime(self):
        if self.value <= 1:
            return False
        for i in range(2, int(self.value/2) + 1):
            if self.value % i == 0:
                return False
        return True

    def ChkPerfect(self):
        if self.value < 2:
            return False
        isum = 0
        for i in range(1, self.value):
            if self.value % i == 0:
                isum = isum + i
        if isum == self.value:
            return True
        else:
            return False

    def factors(self):
        print(f"Factors of are:", end=" ")
        for i in range(2, self.value):
            if self.value % i == 0:
                print(i, end=" ")
        print()

    def SumFactors(self):
        isum = 0
        for i in range(1, self.value):
            if self.value % i == 0:
                isum += i
        return isum

def main():
    num = 0
    print("Enter a number:")
    num = int(input())
    obj = Arithmetic(num)

    iret=obj.ChkPrime()
    if iret==True:
        print("Number is prime") 
    else:
        print("Number is not prime")

    iret= obj.ChkPerfect()
    if iret == True:
        print("Number is a perfect number")
    else:
        print("Number is not a perfect number")

    obj.factors()

    iret2 = obj.SumFactors()
    print("Sum of factors is:", iret2)

if __name__ == "__main__":
    main()
