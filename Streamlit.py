import streamlit as st
import pickle
import string
import time
import spacy

# Page configuration
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="wide")

# Load model and vectorizer
@st.cache_resource(ttl=3600)
def load_model():
    with open('D:\\Spam_Email_Project\\spam_detector.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('D:\\Spam_Email_Project\\count_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Text cleaning functions
def clean_text(s):
    for cs in s:
        if cs not in string.ascii_letters:
            s = s.replace(cs, ' ')
    return s.rstrip('\r\n')

def remove_little(s):
    words_list = s.split()
    k_length = 2
    result_list = [element for element in words_list if len(element) > k_length]
    result_string = ' '.join(result_list)
    return result_string

def lemmatize_text(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text

def preprocess(text):
    return lemmatize_text(remove_little(clean_text(text)))

# Email classification function
def classify_email(model, vectorizer, email):
    prediction = model.predict(vectorizer.transform([email]))
    return prediction

# Main function
def main():
    # Custom CSS for enhanced design
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: White;
            color: White;
        }
        .stApp {
            padding: 2rem;
        }
        .title {
            text-align: center;
            font-weight: bold;
            font-size: 2.5rem;
            color: Black;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666666;
            margin-bottom: 2rem;
        }
        .input-area {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .result-area {
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            text-align: center;
            font-size: 1.1rem;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: White;
        }
        .status-bar {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            margin-top: 10px;
            color: #555555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Page layout
    st.markdown("<h1 class='title'>Spam Email Detector 📧</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Analyze your email content to detect spam instantly.</p>", unsafe_allow_html=True)

    # Input section
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    st.markdown("### 📝 Enter the email text:", unsafe_allow_html=True)
    user_input = st.text_area(
        label="",
        placeholder="e.g., Congratulations! You have won $1,000,000! Click here to claim your prize!",
        height=200
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Button for checking spam
    if st.button("Check for Spam 🚀"):
        if user_input.strip():
            output_placeholder = st.empty()
            status_placeholder = st.empty()

            with status_placeholder.container():
                st.markdown("<div class='status-bar'>Loading the model...</div>", unsafe_allow_html=True)
                time.sleep(1)
                model, vectorizer = load_model()

                st.markdown("<div class='status-bar'>Preprocessing the email content...</div>", unsafe_allow_html=True)
                processed_input = preprocess(user_input)
                time.sleep(1)

                st.markdown("<div class='status-bar'>Analyzing for spam detection...</div>", unsafe_allow_html=True)
                prediction = classify_email(model, vectorizer, processed_input)
                time.sleep(1)

                st.markdown("<div class='status-bar'>Detection completed!</div>", unsafe_allow_html=True)
                time.sleep(0.5)

            # Display results
            status_placeholder.empty()
            st.markdown("<div class='result-area'>", unsafe_allow_html=True)
            if prediction == 1:
                st.error("🚨 **Spam Detected!** This email might be harmful or unwanted.")
            else:
                st.success("✅ **Not Spam!** This email seems safe.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
