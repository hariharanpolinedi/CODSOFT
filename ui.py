import tkinter as tk
from tkinter import messagebox
from joblib import load
import re

# Load the pre-trained model and TFIDF vectorizer
loaded_model = load('movie_genre_classifier.pkl')
loaded_tfidf = load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    # Basic text preprocessing
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_genre(plot_summary):
    clean_summary = preprocess_text(plot_summary)
    features = loaded_tfidf.transform([clean_summary])
    prediction = loaded_model.predict(features)
    return prediction[0]

def on_predict():
    plot_summary = entry.get("1.0", tk.END).strip()
    if plot_summary:
        try:
            predicted_genre = predict_genre(plot_summary)
            messagebox.showinfo("Predicted Genre", f"Predicted Genre: {predicted_genre}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    else:
        messagebox.showwarning("Input Error", "Please enter a plot summary.")

# Create the main window
root = tk.Tk()
root.title("Movie Genre Predictor")

# Create and place the widgets
label = tk.Label(root, text="Enter Movie Plot Summary:")
label.pack(pady=10)

entry = tk.Text(root, height=10, width=50)
entry.pack(pady=10)

predict_button = tk.Button(root, text="Predict Genre", command=on_predict)
predict_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
