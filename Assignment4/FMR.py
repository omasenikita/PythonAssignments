def filterX(Task,Values):
    Result = []
    for No in Values:
        ret = Task(No)
        if(ret == True):
            Result.append(No)
            
    return Result  

def mapX(Task,Values):
    Result = []
    
    for No in Values:
        ret = Task(No) 
        Result.append(ret)
        
    return Result    

def reduceX(Task , Values):
    Result = 0
    for no in Values:
        Result = Task(Result ,no)
        
    return Result