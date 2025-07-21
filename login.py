import streamlit as st
import authlib
from app import chat

IMAGE_ADDRESS = "https://europe.ippf.org/sites/europe/files/styles/header_background_xs/public/2022-12/illustration-14-website-no-logos-no-text.jpg?itok=y-vJsgkS"

# title
st.title("EduBias Detector")

st.image(IMAGE_ADDRESS)
#if not st.experimental_user.is_logged_in:

if not st.user.is_logged_in:
    if st.sidebar.button("Log in with Google", type="primary", icon=":material/login:"):
        st.login()

else:
    #st.html(f"Hello, <span style='color: orange; font-weight: bold;'>{st.experimental_user.name}</span>!")
    if st.sidebar.button("Log out", type="secondary", icon=":material/logout:"):
        st.logout()
    chat()

