
# Streamlit Interface
import streamlit as st
st.set_page_config(page_title="Cover Letter Writter", page_icon='🧾')

with st.sidebar:
    st.header("References")
    st.text_input("Paste Job Post Url")
    file = st.file_uploader("Upload CV File", type=['pdf', 'docx'])
    
    st.button('Generate Cover letter')

# Create a container for the card box
card_box = st.container()

with card_box:
    # Add a title for the card
    st.header("Cover Letter")

    # Add content to the card body
    st.write("This is the card content. You can add text, images, or other elements here.")

    # Add a button or link (optional)
    if st.button("Export"):
        # Perform an action when the button is clicked
        pass

