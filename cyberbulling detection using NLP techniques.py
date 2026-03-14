import streamlit as st
from predict import predict_text

st.title("Cyberbullying Detection System")

st.write("This system detects harmful or bullying comments using NLP.")

comment = st.text_area("Enter a comment")

if st.button("Analyze"):

    if comment.strip() == "":
        st.warning("Please enter a comment")

    else:
        result = predict_text(comment)

        if "Detected" in result:
            st.error(result)
        else:
            st.success(result)