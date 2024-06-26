import streamlit as st
from st_pages import Page, add_page_title, show_pages




def show_home():
    show_pages(
        [
            Page("Home.py", "Home", "🏠"),
            # Can use :<icon-name>: or the actual icon
            Page("pages/1_❄️_Ask_Question.py", "Ask Question", ":books:"),
            # # The pages appear in the order you pass them
            Page("pages/03_Compare_Your_PDF_FAISS.py", "Compare PDFs", "📖"),
            # Page("example_app/example_two.py", "Example Two", "✏️"),
            # # Will use the default icon and name based on the filename if you don't
            # # pass them
            # Page("example_app/example_three.py"),
            # Page("example_app/example_five.py", "Example Five", "🧰"),
        ])

    st.title('Enhanced PDF Analysis Tool')
    st.markdown("-------")
    st.caption("Made with 🖤 by Ziwen Ming", unsafe_allow_html=True)
    # Use a smaller, more relevant banner image


    st.header('Features')
    # Using icons subtly within the text
    st.markdown("""
    🔍 **Ask Questions**: Extract answers from your PDF documents using natural language.
    <br /> 📊 **Compare PDFs**: Compare multiple PDFs to identify key differences and similarities.
    """, unsafe_allow_html=True)
    st.header('Quick Start Guide')
    st.markdown("""
    **Navigate**: Use the sidebar to select between different functionalities.  
    **Upload PDFs**: Upload your documents depending on the task—single PDF for querying or multiple for comparison.  
    **Interact**: Engage with the tool based on prompts to enter questions or initiate comparisons.
        """, unsafe_allow_html=True)



    st.header('Why Use This Tool?')
    st.markdown("""
        This tool is built to support researchers, students, and professionals by simplifying the management and analysis of PDF documents. Whether it's pulling out specific information quickly or comparing textual contents across documents, this tool is designed to save time and enhance productivity.
        """, unsafe_allow_html=True)

    # st.subheader('Technical Overview')
    # with st.expander("Discover More"):
    #     st.write("""
    #     This tool leverages the power of **Natural Language Processing (NLP)**, **Vector Space Modeling**, and **Cosine Similarity** to analyze and compare PDF content:
    #     - **NLP Techniques**: Utilizes state-of-the-art models for extracting and interpreting text.
    #     - **FAISS (Facebook AI Similarity Search)**: Employs this efficient similarity search algorithm for high-dimensional data, facilitating quick and precise document comparisons.
    #     - **Cosine Similarity**: A crucial metric used for measuring the similarity between two documents. By comparing the cosine of the angle between two vectors, which represent document contents, it effectively determines how closely related two documents are in terms of their content.
    #     - **Machine Learning**: Integrates machine learning models to enhance text analysis capabilities, making the tool adaptable to various types of documents.
    #     """)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Technical Overview", expanded=True):
        st.markdown("""
        <h4 style='margin-top:10px; color: #0e1117; font-size: 22px;'>Technical Details:</h4>
        <ul>
            <li><strong>NLP Techniques:</strong> Utilizes state-of-the-art models for extracting and interpreting text.</li>
            <li><strong>FAISS (Facebook AI Similarity Search):</strong> Employs this efficient similarity search algorithm for high-dimensional data, facilitating quick and precise document comparisons.</li>
            <li><strong>Cosine Similarity:</strong> A crucial metric used for measuring the similarity between two documents. By comparing the cosine of the angle between two vectors, which represent document contents, it effectively determines how closely related two documents are in terms of their content.</li>
            <li><strong>Machine Learning:</strong> Integrates machine learning models to enhance text analysis capabilities, making the tool adaptable to various types of documents.</li>
        </ul>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_home()