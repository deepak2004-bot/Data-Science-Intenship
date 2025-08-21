import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.cluster import KMeans

class KMeansApp:
    def __init__(self, root):
        self.root = root
        self.root.title("K-Means Clustering App")
        self.root.geometry("900x700")
        self.root.configure(bg="lightgrey")
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title_frame = tk.Frame(self.root, bg="lightgrey", pady=20)
        title_frame.pack(fill=tk.X)
        title = tk.Label(title_frame, text="K-Means Clustering", font=("bold", 24, "bold"), bg="grey")
        title.pack(pady=10)
        
        # Control Frame
        control_frame = tk.Frame(self.root, bg="lightblue")
        control_frame.pack(pady=10)
        
        # Load Dataset Button
        self.load_button = tk.Button(control_frame, text="Load Dataset", command=self.load_dataset, bg="lightblue", font=("bold", 14))
        self.load_button.grid(row=0, column=0, padx=10, pady=10)
        
        # Elbow Method Button
        self.elbow_button = tk.Button(control_frame, text="Visualize Elbow Method", command=self.visualize_elbow, bg="lightblue", font=("bold", 14))
        self.elbow_button.grid(row=0, column=1, padx=10, pady=10)
        
        # K-Means Button
        self.kmeans_button = tk.Button(control_frame, text="Visualize K-Means Clustering", command=self.visualize_kmeans, bg="lightblue", font=("bold", 14))
        self.kmeans_button.grid(row=0, column=2, padx=10, pady=10)
        
        # Status Bar
        self.status = tk.Label(self.root, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="lightblue", font=("bold", 12))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Figure Canvas
        self.figure_canvas = None
        
    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.dataset = pd.read_csv(file_path)
            self.x = self.dataset.iloc[:, [3, 4]].values
            messagebox.showinfo("Info", "Dataset loaded successfully!")
            self.status.config(text="Status: Dataset loaded")
        else:
            messagebox.showwarning("Warning", "No file selected!")
            self.status.config(text="Status: No file selected")
    
    def visualize_elbow(self):
        if not hasattr(self, 'x'):
            messagebox.showwarning("Warning", "No dataset loaded!")
            self.status.config(text="Status: No dataset loaded")
            return
        
        # Elbow method to find optimal number of clusters
        wcss = []
        for i in range(1, 20):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
            kmeans.fit(self.x)
            wcss.append(kmeans.inertia_)
        
        # Plot Elbow Method
        plt.figure(dpi=300)
        plt.plot(range(1, 20), wcss, marker='o', color='purple', linestyle='--')
        plt.title("The Elbow Method", fontsize=12)
        plt.xlabel("Number of clusters", fontsize=10)
        plt.ylabel("WCSS", fontsize=10)
        plt.grid(True)
        elbow_plot = plt.gcf()
        
        # Remove previous plots if any
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()
        
        # Display the Elbow plot
        self.figure_canvas = FigureCanvasTkAgg(elbow_plot, master=self.root)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.status.config(text="Status: Elbow Method visualized")
    
    def visualize_kmeans(self):
        if not hasattr(self, 'x'):
            messagebox.showwarning("Warning", "No dataset loaded!")
            self.status.config(text="Status: No dataset loaded")
            return
        
        # KMeans with optimal clusters (Assuming 5 here, can be dynamic)
        kmeans = KMeans(n_clusters=5, random_state=0)
        y_kmeans = kmeans.fit_predict(self.x)
        
        # Plot Clusters
        plt.figure(dpi=300)
        plt.scatter(self.x[:, 0], self.x[:, 1], c="lightgrey", label="Data Points")
        plt.scatter(self.x[y_kmeans == 0, 0], self.x[y_kmeans == 0, 1], s=20, c="brown", label="Cluster 1")
        plt.scatter(self.x[y_kmeans == 1, 0], self.x[y_kmeans == 1, 1], s=20, c="yellow", label="Cluster 2")
        plt.scatter(self.x[y_kmeans == 2, 0], self.x[y_kmeans == 2, 1], s=20, c="green", label="Cluster 3")
        plt.scatter(self.x[y_kmeans == 3, 0], self.x[y_kmeans == 3, 1], s=20, c="blue", label="Cluster 4")
        plt.scatter(self.x[y_kmeans == 4, 0], self.x[y_kmeans == 4, 1], s=20, c="pink", label="Cluster 5")
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="red", label="Centroids")
        plt.title("Clusters of Customers", fontsize=12)
        plt.xlabel("Annual Income", fontsize=10)
        plt.ylabel("Spending Score", fontsize=10)
        plt.legend()
        plt.grid(True)
        clusters_plot = plt.gcf()
        
        # Remove previous plots if any
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()
        
        # Display the K-Means clustering plot
        self.figure_canvas = FigureCanvasTkAgg(clusters_plot, master=self.root)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.status.config(text="Status: K-Means Clustering visualized")

# Create the Tkinter root window and run the application
root = tk.Tk()
app = KMeansApp(root)
root.mainloop()
