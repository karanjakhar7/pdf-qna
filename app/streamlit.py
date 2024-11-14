import streamlit as st
from core.pipeline import create_qa_chain

st.set_page_config(page_title="PDF Q&A Assistant", page_icon="📚", layout="centered")

st.title("PDF Q&A Assistant")

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if not uploaded_file and st.session_state.qa_chain is not None:
    st.session_state.qa_chain = None
    st.rerun()

if uploaded_file:
    # Only process PDF if we haven't already
    if st.session_state.qa_chain is None:
        with st.spinner("Processing PDF..."):
            import os
            import tempfile

            # Create a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)

            # Save uploaded file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Create QA chain and store in session state
            st.session_state.qa_chain = create_qa_chain(temp_path)

            st.success("PDF processed successfully!")

    # Question input
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            # Get answer using the stored chain
            answer = st.session_state.qa_chain.invoke(question)

            # Display answer
            st.write("Answer:")
            st.write(answer)

else:
    st.info("Please upload a PDF file to begin.")
