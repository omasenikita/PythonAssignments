
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def StudentDetails():
    data = {
        'Name':['Amit','Sagar','Pooja'],
        'Math':[85,90,78],
        'Science':[92,88,82],
        'English':[75,85,82]
    }
   
   #converting the dictionary to dataframe
    print(data)
    
    print("Converting the dictionary to dataframe")
    df = pd.DataFrame(data)
     
    
    
    #Adding a new column Row_Total
    print("Add Gender column")
    df['Gender'] = ['M', 'M', 'F'];
    
    print(df)
    print("One Hot Encoding")
    
    
    
    # unique names
    df['Name'].unique()
    print("Unique names:", df['Name'].unique()) 
    
    
    
     
    # one hot encoding on Gender column
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    name_encoded = ohe.fit_transform(df[['Gender']])
    print(name_encoded)
    
    df['Row_Total'] = df['Math']+df['Science']+df['English']



    
    #Groupp by Gender and calculate Avearage marks
    average_marks_by_gender = df.groupby('Gender')['Row_Total'].mean()
    print("Average marks:",average_marks_by_gender)
    
    
    
    
    #plot a pie Plot for Subject Marks for Sagar
    marks = df[df['Name']=='Sagar'][['Math','Science','English']].values.flatten()
    subjects = ['Math','Science','English']
    plt.pie(marks, labels=subjects, autopct='%1.1f%%', startangle=140)
    plt.title("Sagar's Subject Marks Pie plot")
    plt.axis('equal')  
    plt.show()
    
    
    
    # Adding a new column Status based on Row_Total
   # Apply if condition using lambda function
    df['Status'] = df['Row_Total'].apply(lambda x: 'Pass' if x >= 250 else 'Fail')    
    
    print("Dataframe with Status column:")
    print(df.head())
    
    
    
    #Count the number of students who passed 
    pass_count = df[df['Status'] == 'Pass'].shape[0]
    print(f"Number of students who passed: {pass_count}")
    
    
    
    #Comvert the dataframe to csv file
    df.to_csv('student_data1.csv', index=False)
    print("Dataframe saved to student_data.csv")
    
    
    
    
    #Applying min max scaling to Math column
    scaler = MinMaxScaler()
    scalemath = scaler.fit_transform(df[['Math']])
    print("After scaling Math marks:")
    
    
    
    
    #calculate Histogram for Math marks
    plt.hist(df['Math'],bins = 5,edgecolor='black')
    plt.xlabel('Math Marks')
    plt.ylabel('Frequency')
    plt.title('Histogram of Math Marks')
    plt.show()
    
    
    
    #rename math column to mathematics
    df.rename(columns={'Math': 'Mathematics'}, inplace=True)
    print("Dataframe after renaming Math to Mathematics:")
    print(df.head())
    
    
    
    #plot box plot for English marks
    plt.boxplot(df['English'])
    plt.title('Box Plot of English Marks')
    plt.ylabel('Marks')
    plt.xlabel('English')
    plt.show()
    
def main():
    StudentDetails()
    
    
if __name__ == "__main__":
    main()
