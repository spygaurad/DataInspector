import os
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()


def simple_chat(
    question: str,
    context: str,
    image_paths: Optional[List[str]] = None
) -> str:
    """
    Simple chat function that answers questions based on context and optional images.
    
    Args:
        question: User's question
        context: Context information (e.g., dataset summary, analysis results)
        image_paths: Optional list of image file paths to include
        
    Returns:
        AI response as a string
    """
    try:
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )
        
        # Build the prompt
        prompt = f"""You are a helpful data analysis assistant.

Context:
{context}

User Question: {question}

Please provide a clear, concise answer based on the context provided."""
        
        # Handle images if provided
        if image_paths:
            content = [{"type": "text", "text": prompt}]
            
            for img_path in image_paths:
                if os.path.exists(img_path):
                    import base64
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode()
                    
                    # Determine image type
                    ext = os.path.splitext(img_path)[1].lower()
                    mime_type = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/png')
                    
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:{mime_type};base64,{img_data}"
                    })
            
            message = HumanMessage(content=content)
        else:
            message = HumanMessage(content=prompt)
        
        # Get response
        response = llm.invoke([message])
        
        # Extract text from response
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)
        
    except Exception as e:
        return f"Error: {str(e)}"


# Example usage:
if __name__ == "__main__":
    # Simple text-only example
    context = """
    Dataset: Customer Sales Data
    - 1000 rows, 15 columns
    - Label: purchase_made (binary)
    - Task: Classification
    - Missing values: 5% in age column
    """
    
    question = "What's the main task for this dataset?"
    response = simple_chat(question, context)
    print(response)
    
    # With images
    question2 = "What do you see in the visualization?"
    response2 = simple_chat(question2, context, image_paths=["/path/to/plot.png"])
    print(response2)