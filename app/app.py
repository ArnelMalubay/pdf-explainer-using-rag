# PDF Explainer Chatbot - Upload PDFs and ask questions about their content

import gradio as gr
from llm import chat_with_assistant_rag, SYSTEM_MESSAGE
from retrieval import access_chroma_collection, parse_pdf, add_documents

# Global collection name
COLLECTION_NAME = "pdf_collection"

# Function to handle PDF uploads
def handle_pdf_upload(files):
    """
    Process uploaded PDF files and add them to the Chroma collection
    """
    if not files:
        return "No files uploaded."
    
    try:
        processed_files = []
        for file in files:
            # Parse the PDF
            pages = parse_pdf(file.name)
            if pages:
                # Add documents to collection
                add_documents(COLLECTION_NAME, pages)
                processed_files.append(file.name.split('/')[-1])  # Get filename only
        
        if processed_files:
            file_list = ", ".join(processed_files)
            return f"✅ Successfully processed and indexed: {file_list}. The documents are now available for questions!"
        else:
            return "❌ Failed to process the uploaded files. Please check the file format."
            
    except Exception as e:
        return f"❌ Error processing files: {str(e)}"

# Main chat function that handles the conversation
def respond(message, history):
    """
    Handle user messages and return streaming responses with RAG
    """
    if not message.strip():
        yield "Please enter a message."
        return
    
    # Get the streaming generator and yield each response
    for partial_response in chat_with_assistant_rag(message, history, COLLECTION_NAME):
        yield partial_response

# Create the chatbot interface with file upload
with gr.Blocks(title = "PDF Explainer Chatbot") as demo:
    gr.Markdown("# 📄 PDF Explainer Chatbot")
    gr.Markdown("""
    **I'm an AI assistant that can help you with general questions and analyze PDF documents you upload.**
    
    - 💬 **Chat normally**: Ask me anything, even without uploading PDFs
    - 📤 **Upload PDFs**: Add documents anytime to get document-specific answers  
    - 🔄 **Multiple uploads**: You can upload more PDFs during our conversation
    - 🎯 **Smart retrieval**: I'll automatically find relevant content from your PDFs when answering questions
    """)
    
    # File upload component
    with gr.Row():
        file_upload = gr.File(
            label = "📄 Upload PDF Documents (Optional)",
            file_count = "multiple",
            file_types = [".pdf"],
            type = "filepath",
            height = 100
        )
        upload_button = gr.Button("🚀 Process PDFs", variant = "primary", size = "sm")
    
    # Upload status
    upload_status = gr.Textbox(label = "Upload Status", interactive = False, visible = False)
    
    # Chat interface
    chatbot = gr.ChatInterface(
        fn = respond,
        type = "messages",
        title = "💬 Chat",
        description = "Ask me anything! If you've uploaded PDFs, I'll use them to provide more accurate answers."
    )
    
    # Handle file upload
    def show_status_and_process(files):
        result = handle_pdf_upload(files)
        return result, gr.update(visible = True)
    
    upload_button.click(
        fn = show_status_and_process,
        inputs = [file_upload],
        outputs = [upload_status, upload_status]
    )

if __name__ == "__main__":
    # Initialize the Chroma collection
    collection = access_chroma_collection(COLLECTION_NAME)
    print(f"✅ Initialized collection: {COLLECTION_NAME}")
    
    # Enable queuing for streaming support
    demo.queue().launch(server_name = "0.0.0.0", server_port = 7860)