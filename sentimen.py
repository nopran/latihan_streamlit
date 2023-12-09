import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Fungsi untuk memuat model dan vectorizer
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
        model = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Fungsi untuk membuat prediksi
def predict_sentiment(model, vectorizer, review):
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return prediction[0]

# Fungsi untuk memuat data (sesuaikan dengan data Anda)
def load_data():
    data = pd.read_csv('dataset-lazada-reviews.csv')
    return data

# Fungsi untuk menampilkan grafik distribusi rating
def plot_rating_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='rating', data=data, palette='viridis')
    plt.title('Distribusi Rating Produk', fontsize=15)
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Jumlah Review', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt.gcf())

# Fungsi untuk analisis deskriptif dan grafik
def show_descriptive_analysis(data):
    st.write("### Analisis Deskriptif Data")
    st.write(data.describe())

    # Menampilkan grafik distribusi rating
    plot_rating_distribution(data)

    # Tambahkan lebih banyak grafik sesuai kebutuhan

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Klasifikasi Sentimen Review Produk")

    # Menu navigasi dengan ikon
    menu_options = {
        "Beranda": "house",
        "Dataset": "book",
        "Model": "gear",
        "Klasifikasi Sentimen": "chat-left-text",
        "Analisis Deskriptif": "bar-chart-line",
        "Tentang": "info-circle"
    }

    st.sidebar.title("Navigasi")
    choice = st.sidebar.radio("", list(menu_options.keys()), format_func=lambda x: f"{menu_options[x]} {x}")

    if choice == "Beranda":
        st.subheader("Beranda")
        st.write("Selamat datang di aplikasi klasifikasi sentimen!")
    elif choice == "Dataset":
        st.subheader("Dataset")
        st.write("Deskripsi dan informasi mengenai dataset yang digunakan.")
        data = load_data()
        st.write(data.head())
    elif choice == "Model":
        st.subheader("Model")
        st.write("Informasi mengenai model yang digunakan untuk klasifikasi sentimen.")
    elif choice == "Klasifikasi Sentimen":
        st.subheader("Klasifikasi Sentimen")
        model, vectorizer = load_model_and_vectorizer()
        review_input = st.text_area("Masukkan Review Produk:")
        if st.button("Klasifikasi"):
            sentiment = predict_sentiment(model, vectorizer, review_input)
            st.write(f"Sentimen Prediksi: {sentiment}")
    elif choice == "Analisis Deskriptif":
        st.subheader("Analisis Deskriptif")
        data = load_data()
        show_descriptive_analysis(data)
    elif choice == "Tentang":
        st.subheader("Tentang Aplikasi")
        st.write("Aplikasi ini dikembangkan untuk mengklasifikasikan sentimen dari review produk.")

# Memanggil fungsi utama
if __name__ == '__main__':
    main()
