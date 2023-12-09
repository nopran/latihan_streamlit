import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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

# Fungsi untuk memuat data
def load_data():
    data = pd.read_csv('dataset-lazada-reviews.csv')
    return data


# Fungsi untuk menampilkan word cloud dari review
def plot_word_cloud(data):
    text = " ".join(str(review) for review in data.reviewContent)
    wordcloud = WordCloud(background_color='white').generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt.gcf())

def show_descriptive_analysis(data):
    st.write("### Analisis Deskriptif Data")
    st.write(data.describe())
    plot_word_cloud(data)

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Klasifikasi Sentimen Review Produk")

    menu_options = {
        "Beranda": ":house:",
        "Dataset": ":book:",
        "Model": ":gear:",
        "Klasifikasi Sentimen": ":memo:",
        "Analisis Deskriptif": ":bar_chart:",
        "Tentang": ":info:"
    }

    st.sidebar.title("Navigasi")
    choice = st.sidebar.radio("", list(menu_options.keys()), format_func=lambda x: f"{menu_options[x]} {x}")

    if choice == "Beranda":
        st.subheader("Beranda")
        st.write("Selamat datang di Portofolio Analisis Review Sentimen!")
    elif choice == "Dataset":
        st.subheader("Dataset")
        st.write("Dataset yang digunakan adalah dataset Lazada Reviews.")
        data = load_data()
        st.write(data.head())
    elif choice == "Model":
        st.subheader("Model")
        st.write("Informasi mengenai model yang digunakan untuk klasifikasi sentimen.")
        # Anda dapat menambahkan lebih banyak informasi model di sini
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
