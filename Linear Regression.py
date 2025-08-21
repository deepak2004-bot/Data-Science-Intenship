import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# Global variables
df = pd.DataFrame()
X = None
y = None
indep_selected = []
scaler = None
regressor = None
regression_type = None

# Function to load the dataset
def load_dataset():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Dataset Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showerror("Error", "Failed to load dataset")

# Function to go to the regression type selection page
def go_to_regression_type_page():
    root.withdraw()
    reg_type_window = tk.Toplevel(root)
    reg_type_window.title("Select Regression Type")
    reg_type_window.geometry("800x600")
    reg_type_window.configure(bg='lightblue')  # Changed background color

    tk.Label(reg_type_window, text="Select Regression Type:", bg='blue', fg='black').pack(pady=60)  # Changed text color

    # Combo box for regression type
    reg_type_combobox = ttk.Combobox(reg_type_window, values=["Simple Linear Regression", "Multiple Linear Regression"], state="readonly")
    reg_type_combobox.pack(pady=40)

    def on_combobox_select(event):
        global regression_type
        regression_type = reg_type_combobox.get()
        reg_type_window.destroy()
        go_to_enter_variables_page()

    reg_type_combobox.bind("<<ComboboxSelected>>", on_combobox_select)

# Function to go to the enter variables page
def go_to_enter_variables_page():
    enter_vars_window = tk.Toplevel(root)
    enter_vars_window.title("Enter Variables")
    enter_vars_window.geometry("800x600")
    enter_vars_window.configure(bg='lightblue')  # Changed background color

    tk.Label(enter_vars_window, text="Enter independent variables (comma separated):", bg='lightgray', fg='black').pack(pady=40)  # Changed text color
    indep_entry = tk.Entry(enter_vars_window, width=50, bg='lightgray')  # Changed entry background color
    indep_entry.pack(pady=5)

    tk.Label(enter_vars_window, text="Enter dependent variable:", bg='lightgray', fg='black').pack(pady=40)  # Changed text color
    dep_entry = tk.Entry(enter_vars_window, width=50, bg='lightgray')  # Changed entry background color
    dep_entry.pack(pady=5)

    def set_variables():
        selected_indep = [var.strip() for var in indep_entry.get().split(',')]
        selected_dep = dep_entry.get().strip()

        global X, y, indep_selected
        try:
            if regression_type == "Simple Linear Regression" and len(selected_indep) != 1:
                messagebox.showerror("Error", "For Simple Linear Regression, only one independent variable is allowed.")
                return
            
            if regression_type == "Multiple Linear Regression" and len(selected_indep) < 2:
                messagebox.showerror("Error", "For Multiple Linear Regression, at least two independent variables are required.")
                return

            X = df[selected_indep].values
            y = df[selected_dep].values
            indep_selected = selected_indep
            messagebox.showinfo("Info", "Variables Set Successfully!")
            enter_vars_window.destroy()
            go_to_train_model_page()
        except KeyError:
            messagebox.showerror("Error", "Invalid variable names. Please check and try again.")

    tk.Button(enter_vars_window, text="Set Variables", command=set_variables, bg='blue', fg='black').pack(pady=40)  # Changed button color

# Function to go to the training model page
def go_to_train_model_page():
    train_model_window = tk.Toplevel(root)
    train_model_window.title("Train Model")
    train_model_window.geometry("800x600")
    train_model_window.configure(bg='lightblue')  # Changed background color

    def train_model():
        try:
            global scaler, regressor, X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            regressor = LinearRegression()
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)
            score = regressor.score(X_test, y_test)
            cross_val_scores = cross_val_score(regressor, X_train, y_train, cv=5)

            result_text = (
                f"Model Trained Successfully!\n\n"
                f"R^2 Score: {score:.2f}\n"
                f"Cross-Validation Scores: {cross_val_scores}\n\n"
                f"Predictions on Test Set:\n{y_pred}\n\n"
                f"Actual Values:\n{y_test}"
            )
            messagebox.showinfo("Model Result", result_text)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while training the model: {e}")

    tk.Button(train_model_window, text="Train Model", command=train_model, bg='lightgray', fg='black').pack(pady=40)  # Changed button color
    tk.Button(train_model_window, text="Go to Prediction Page", command=go_to_prediction_page, bg='lightgray', fg='black').pack(pady=40)  # Changed button color

# Function to go to the prediction page
def go_to_prediction_page():
    prediction_window = tk.Toplevel(root)
    prediction_window.title("Enter Input Values")
    prediction_window.geometry("800x600")
    prediction_window.configure(bg='lightblue')  # Changed background color

    inputs = []

    for feature in indep_selected:
        frame = tk.Frame(prediction_window, bg='lightgray')  # Changed frame background color
        frame.pack(pady=5)
        label = tk.Label(frame, text=f"Enter value for {feature}:", bg='lightgray', fg='black')  # Changed label color
        label.pack(side=tk.LEFT, padx=5)
        entry = tk.Entry(frame, bg='lightblue')  # Changed entry background color
        entry.pack(side=tk.RIGHT, padx=5)
        inputs.append(entry)

    def predict():
        try:
            input_values = [float(entry.get()) for entry in inputs]
            input_array = np.array(input_values).reshape(1, -1)
            input_array = scaler.transform(input_array)  # Scale input data
            prediction = regressor.predict(input_array)
            messagebox.showinfo("Prediction", f"Predicted Value: {prediction[0]}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {e}")

    tk.Button(prediction_window, text="Predict", command=predict, bg='lightgray', fg='black').pack(pady=20)  # Changed button color

# Main window
root = tk.Tk()
root.title("Machine Learning Deployment")
root.geometry("800x600")
root.configure(bg='lightblue')  # Changed background color

tk.Button(root, text="Load Dataset", command=load_dataset, bg='lightgray', fg='black').pack(pady=40)  # Changed button color
tk.Button(root, text="Next", command=go_to_regression_type_page, bg='lightgray', fg='black').pack(pady=40)  # Changed button color

root.mainloop()
