import pandas as pd
import streamlit as st
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

st.markdown("""
<style>
.stButton > button {
    background-color: none;
    color: white;
    padding: 10px 20px;
    border-radius: 10px;
    border: 2px solid white;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
    width: 200px;
    text-align: center;
    display: block;
    margin: 0 auto;
}

.stButton > button:hover {
    background-color: #FA8072;
    border: none;
    box-shadow: 0 0px 5px pink;
    transform: translateY(-2px);
    color: black;
    font-size: 24px;
}

.stButton > button:active {
    transform: translateY(0px);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Employee Salary Predictor with Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("employee_salaries_csv.csv")

df = load_data()
st.subheader("📄 Raw Dataset")
st.dataframe(df,height=200)

def preprocess(df):
    df = df.copy()
    encoders = {}

    for col in ["Education Level", "Department", "Location"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

df_encoded, encoders = preprocess(df)

X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "salary_model.pkl")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.success(f"✅ Model trained. Mean Squared Error: {mse:,.2f}")


st.divider()

st.header("🔍 Predict Salary")

experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=1)
education = st.selectbox("Education Level", encoders["Education Level"].classes_)
department = st.selectbox("Department", encoders["Department"].classes_)
location = st.selectbox("Location", encoders["Location"].classes_)

if st.button("Predict Salary"):
    input_data = pd.DataFrame({
        "Experience": [experience],
        "Education Level": encoders["Education Level"].transform([education]),
        "Department": encoders["Department"].transform([department]),
        "Location": encoders["Location"].transform([location])
    })
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: ${prediction:,.2f}")

st.divider()

st.header("📊 Data Visualizations")

col1, col2, col3 = st.columns(3)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Salary"], kde=True, ax=ax1)
    plt.xticks(rotation=45)
    plt.title("Salary Distribution")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.barplot(data=df, x="Department", y="Salary", ax=ax2, color='#FFD700')
    plt.xticks(rotation=45)
    plt.title("Avg Salary by Department")
    st.pyplot(fig2)

with col3:
    fig3, ax3 = plt.subplots()  
    sns.boxplot(data=df, x="Education Level", y="Salary", ax=ax3 ,color="#974DE1")
    plt.xticks(rotation=45)  # Smaller font size
    plt.title("Salary by Education")
    st.pyplot(fig3)

st.divider()

col4, col5, col6 = st.columns(3)

with col4:
    fig4, ax4 = plt.subplots()
    sns.scatterplot(data=df, x="Experience", y="Salary", ax=ax4 ,color='lime', alpha=0.6, s=100)
    plt.xticks(rotation=45)
    plt.title("Salary vs Experience")
    st.pyplot(fig4)

with col5:
    fig5, ax5 = plt.subplots()
    sns.pointplot(data=df, x="Location", y="Salary", ax=ax5, color='#B22222')
    plt.xticks(rotation=45)
    plt.title("Avg Salary by Location")
    st.pyplot(fig5)

with col6:
    fig6, ax6 = plt.subplots()
    correlation = df_encoded.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax6)
    plt.title("Correlation Heatmap")
    st.pyplot(fig6)

st.divider()
