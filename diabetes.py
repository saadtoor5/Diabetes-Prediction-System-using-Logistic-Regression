#Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pylab as plt

#Loading Pima Indian Diabetes Dataset
data = pd.read_csv('diabetes.csv')
data.head()
X = data.drop('Outcome', axis=1)
y = data['Outcome']

#Splitting data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Training model on Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

#Making prediction on the test data
y_pred = model.predict(X_test)

#Evaluation
accuracy = accuracy_score(y_test, y_pred)
con_mat = confusion_matrix(y_test,y_pred)
class_rep = classification_report(y_test,y_pred)

#Visualization
print(f"Accuracy Score: {accuracy}\nConfusion Matrix: {con_mat}\nClassification Report: {class_rep}")
sns.heatmap(con_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Function to predict using user input
def user_input_data(pg, gl, bp, st, il, bm, dp, ag):
    new_data = pd.DataFrame({
        'Pregnancies': [pg],
        'Glucose': [gl],
        'BloodPressure': [bp],
        'SkinThickness': [st],
        'Insulin': [il],
        'BMI': [bm],
        'DiabetesPedigreeFunction': [dp],
        'Age': [ag]
    })

    predicted_result = model.predict(new_data)
    return predicted_result


#Function to get the data input from the user into a proper format
def get_user_input():
    print("Please enter the following details:\n")

    try:
        pg = int(input("🍼 Pregnancies (0 or more): "))
        gl = int(input("🩸 Glucose Level (e.g., 85–200): "))
        bp = int(input("💓 Blood Pressure Level (mm Hg): "))
        st = int(input("📏 Skin Thickness (mm): "))
        il = int(input("💉 Insulin Level (mu U/ml): "))
        bm = float(input("⚖️ BMI (e.g., 18.5–50.0): "))
        dp = float(input("🧬 Diabetes Pedigree Function (e.g., 0.2–2.5): "))
        ag = int(input("🎂 Age (years): "))

        return pg, gl, bp, st, il, bm, dp, ag

    except ValueError:
        print("\n❌ Invalid input detected. Please enter numeric values only.\n")
        return get_user_input()
    
# Collect and predict user input
pg, gl, bp, st, il, bm, dp, ag = get_user_input()
result = user_input_data(pg, gl, bp, st, il, bm, dp, ag)

print(f"\n🔍 Predicted Result: {'🟥 Diabetic' if result == 1 else '🟩 Non Diabetic'}")