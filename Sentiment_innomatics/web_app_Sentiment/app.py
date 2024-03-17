import streamlit as st
import joblib
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import unicodedata
import nltk
import plotly.express as px

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load("C:\\Users\\svani\\Sentiment_innomatics\\model\\Innomatics_batmiton_sentiment_model.joblib")

def clean_text(sentence):
    text = sentence.lower()

    text = BeautifulSoup(text, 'html.parser').get_text()

    text = ' '.join([word for word in text.split() if not word.startswith('http')])

    text = ''.join([char for char in text if char not in string.punctuation + '’‘'])

    text = ''.join([i for i in text if not i.isdigit()])

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    text = ' '.join([word for word in word_tokens if word.lower() not in stop_words])

    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])

    return text

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])
    return prediction[0]
def main():
    st.image("https://s3-ap-southeast-1.amazonaws.com/tv-prod/member/photo/4353063-large.png")

    st.title("Sentiment Analysis for Brand Reputation🤓")

    st.sidebar.header("Upload File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "txt", "json", "xlsx", "xls"])

    if uploaded_file is not None:
        df = None
        try:
            if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")

        if df is not None:
            st.subheader("Preview of Uploaded Data")
            st.write(df.head())

            st.subheader("Sentiment Prediction for Text Column")
            text_column = st.selectbox("Select the text column for sentiment analysis", df.columns)

            predictions = df[text_column].apply(predict_sentiment)

            df['Predicted Sentiment'] = predictions
            st.write(df[[text_column, 'Predicted Sentiment']])

            st.subheader("Bar Chart of Predicted Sentiments 📊")
            fig = px.bar(df, x='Predicted Sentiment', title='Predicted Sentiments Distribution')
            st.plotly_chart(fig, use_container_width=True)

            
            st.subheader("Pie Chart 🥧🍰")
            pie_fig = px.pie(df, names='Predicted Sentiment', title='Predicted Sentiments Distribution (Pie Chart)')
            st.plotly_chart(pie_fig, use_container_width=True)

            positive_count = (df['Predicted Sentiment'] == 'positive').sum()
            negative_count = (df['Predicted Sentiment'] == 'negative').sum()

            if positive_count > negative_count:
                st.title("Wow You reached More Positive Reviews")
                st.markdown('<div style="text-align: center;"><img src="https://emojicdn.elk.sh/😊"></div>', unsafe_allow_html=True)

            elif positive_count < negative_count:
                st.title("Negative 🥲,Dont worry, Great things take Time✌️")
                st.markdown('<div style="text-align: center;"><img src="https://emojicdn.elk.sh/😞"></div>', unsafe_allow_html=True)

            else:
                st.write("Both positive and negative sentiments have equal counts.")

    else:
        user_input = st.text_area("Enter a review text:")

        if st.button("Predict"):
            if user_input:
                sentiment = predict_sentiment(user_input)
                if sentiment == "positive":
                    st.markdown("<h2 style='text-align: center;'>Positive Sentiment Detected!</h2>", unsafe_allow_html=True)
                    st.markdown('<div style="text-align: center;"><img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExeXg2N3hxaGh5ZmpkdGJkNzhzM2NyNnZnMjVlZjhrcnZrbHlweGJpMCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/chzz1FQgqhytWRWbp3/giphy.webp"></div>', unsafe_allow_html=True)
                elif sentiment == "negative":
                    st.markdown("<h2 style='text-align: center;'>Negative Sentiment Detected!</h2>", unsafe_allow_html=True)
                    st.markdown('<div style="text-align: center;"><img src="https://media0.giphy.com/media/l1KVaj5UcbHwrBMqI/200.webp?cid=790b7611iax98h1kpxexlwp7nntp5pf4pknw720hkr687ep7&ep=v1_gifs_search&rid=200.webp&ct=g"></div>', unsafe_allow_html=True)
                else:
                    st.error("Sentiment could not be determined.")
            else:
                st.warning("Please enter a review text.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon=":smiley:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
            body {
                background-color: FFFF00;
                color: #FFFF00;   
                font-size:20px; 
            }
            .stTextInput, .stTextArea {
                background-color: FFFF00;  
                color: #FFFF00;  
                font-size:200px;
  
            }
            .stButton>button {
                background-color: #4682B4;
                color: #000000;
            }
            .sttitle{
            font-color:red;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    main()
