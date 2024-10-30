import streamlit as st
import cv2
import numpy as np
from index import process_ndvi
import tempfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_chat import message
import requests

st.set_page_config(page_title="RUSERO Analysis Tools", layout="wide")

# Initialize chat history in session state if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def handle_chat_input(user_input):
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Prepare the request to your FastAPI backend
        try:
            response = requests.post(
                "http://localhost:8001/chat",  # Update with your API endpoint
                data={
                    "message": user_input,
                    "user_id": "web-user"
                }
            )
            ai_response = response.json()["response"]

            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        except Exception as e:
            st.error(f"Error communicating with chat service: {str(e)}")


def plot_reflectance_distribution(reflectance):
    fig, ax = plt.subplots(figsize=(10, 4))

    # Calculate percentage instead of raw frequency
    total_pixels = reflectance.size
    sns.histplot(reflectance.flatten(), bins=50, ax=ax, stat='percent')

    ax.set_title('Distribution of NIR Reflectance Values')
    ax.set_xlabel('NIR Reflectance Value (0-255)')
    ax.set_ylabel('Percentage of Image Pixels (%)')

    # Add some explanatory text
    plt.figtext(0.02, -0.15,
                "This histogram shows how NIR reflectance values are distributed across the image.\n" +
                "Higher values (towards 255) indicate stronger NIR reflection, often associated with dense vegetation.",
                wrap=True)

    return fig


def mapping_tool():
    st.title("🗺️ Mapping Tool")
    st.write("Coming soon! This feature will allow you to visualize geographical data.")

def ndvi_analysis():
    st.title("🌿 RUSERO NDVI Analysis Tool")
    st.write("Upload an image from a NoIR camera with blue filter for NDVI analysis")

def ndvi_analysis():
    st.title("🌿 RUSERO NDVI Analysis Tool")
    st.write("Upload an image from a NoIR camera with blue filter for NDVI analysis")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            # Create temporary output path
            output_path = temp_path.replace('.jpg', '_output.jpg')
            
            # Process the image
            stats = process_ndvi(temp_path, output_path)

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("NDVI Analysis Results")
                st.image(output_path, caption="NDVI Analysis", use_column_width=True)
                
                # Display individual images
                base_name = output_path.rsplit('.', 1)[0]
                st.image(f"{base_name}_reflectance.jpg", caption="NIR Reflectance", use_column_width=True)
                st.image(f"{base_name}_ndvi.jpg", caption="NDVI Colormap", use_column_width=True)

            with col2:
                st.subheader("Statistics")
                # Create a nice looking metrics display
                st.metric("Average NIR Reflectance", f"{stats['average_reflectance']:.1f}")
                st.metric("Vegetation Coverage", f"{stats['vegetation_coverage'] * 100:.1f}%")
                    

                # Plot reflectance distribution
                st.subheader("Reflectance Distribution")
                img = cv2.imread(temp_path)
                b, _, _ = cv2.split(img)
                reflectance = cv2.normalize(b.astype(float), None, 0, 255, cv2.NORM_MINMAX)
                fig = plot_reflectance_distribution(reflectance)
                st.pyplot(fig)

                # Add interpretation guide
                st.subheader("NDVI Interpretation Guide")
                st.write("""
                - NDVI values range from -1 to 1
                - Values > 0.2: Indicates presence of vegetation
                - Values > 0.5: Indicates dense/healthy vegetation
                - Values < 0.1: Indicates non-vegetated areas
                """)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        finally:
            # Cleanup temporary files
            os.unlink(temp_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
                os.unlink(f"{base_name}_reflectance.jpg")
                os.unlink(f"{base_name}_ndvi.jpg")

    
def main():
    # Create main content area and sidebar
    # main_content, chat_sidebar = st.columns([0.7, 0.3])
     # Navigation tabs
        tab1, tab2, tab3 = st.tabs(["NDVI Analysis", "Mapping", "Ai Assist"])
        
        with tab1:
            ndvi_analysis()
            
        with tab2:
            mapping_tool()
        
        with tab3:
            st.markdown("### 💬 Chat Assistant")
            
            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for chat in st.session_state.chat_history:
                    message(
                        chat["content"],
                        is_user=chat["role"] == "user",
                        key=str(hash(chat["content"]))
                    )
            
            # Chat input
            st.markdown("---")
            user_input = st.text_input("Ask a question:", key="chat_input")
            if st.button("Send", key="send_button"):
                handle_chat_input(user_input)
                # Clear input after sending
                st.session_state.chat_input = ""
       

if __name__ == "__main__":
    main()
    
