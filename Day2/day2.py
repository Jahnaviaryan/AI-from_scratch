from sklearn.linear_model import LinearRegression as lg
 #data
experience = [[1], [2], [3], [4]]
salary = [20000, 30000, 40000, 50000]

#model
model = lg()

#training 
model.fit(experience, salary)

#prediction
print(model.predict([[5]]))


 
#input from user

from sklearn.linear_model import LinearRegression

# data
hours = [[1], [2], [3], [4]]
marks = [30, 50, 65, 80]

# model
model = LinearRegression()
model.fit(hours, marks)

# user input
h = float(input("Enter study hours: "))

# prediction
pred = model.predict([[h]])

print("Predicted marks:", pred[0])






# view m and b

from sklearn.linear_model import LinearRegression

x = [[1], [2], [3], [4]]
y = [10, 20, 25, 40]


model = LinearRegression()
model.fit(x, y)

print("Prediction for 5:", model.predict([[5]]))
print("m:", model.coef_)
print("b:", model.intercept_)







#LogisticRegression


from sklearn.linear_model import LogisticRegression

# Step 1: Data
x = [[1], [2], [3], [4], [5]]
y = [0, 0, 0, 1, 1]   # 0 = Fail, 1 = Pass          #input data

# Step 2: Model
model = LogisticRegression()                          #decide model
model.fit(x, y)                                        #fits

# Step 3: User Input
hours = float(input("Enter study hours: "))                #user inputs and stored in hours

# Step 4: Prediction
result = model.predict([[hours]])                       #prdiction storing

# Step 5: Output
if result[0] == 1:
    print("Result: PASS ✅")                               #rules in if
else:
    print("Result: FAIL ❌")




