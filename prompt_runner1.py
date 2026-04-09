import os
import google.generativeai as genai
import dotenv

dotenv.load_dotenv()
def main():
    # 1. Set up your API key (Replace '1234-abc' with your actual key)
    # Best practice is to set this as an environment variable, but hardcoding for the example.
    api_key = os.getenv("GEMINI_API")  # Replace with your actual API key or set as env variable
    genai.configure(api_key=api_key)

    # 2. Read your system prompt from the local file
    try:
        with open("prompts/system_prompt.md", "r", encoding="utf-8") as f:
            system_instruction = f.read()
    except FileNotFoundError:
        print("Error: system_prompt.md not found. Please ensure it's in the same directory.")
        return

    # 3. Upload your PDF slides to the Gemini File API
    # The File API is necessary for handling documents like PDFs effectively.
    print("Uploading PDF slides...")
    try:
        slide1 = genai.upload_file("/home/austra/Downloads/CSE 329 Assignment/XGBoost Lectures/lecture1.pdf")
        print("Uploads complete.")
    except Exception as e:
        print(f"Error uploading files: {e}")
        return

    # 4. Initialize the model
    # gemini-2.5-pro is the recommended model for complex reasoning and large document analysis
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=system_instruction
    )

    # 5. Define the user prompt explicitly tying the slides to the task
    user_prompt = (
        "Please generate the module-specific lecture content for 'Ensembles of Trees (XGBoost)'. "
        "It is critical that you base this content STRICTLY on the material, concepts, and flow "
        "provided in the attached slides. Do not invent new concepts that aren't covered in the slides, "
        "but ensure you meet all pedagogical and formatting requirements outlined in the system instructions."
    )

    # 6. Generate the content passing the file objects AND the text prompt
    print("Generating lecture content... (this may take a moment depending on slide length)")
    response = model.generate_content([slide1, user_prompt])

    # 7. Save the output to a Markdown file
    output_filename = "xgboost_lecture_content.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    print(f"Success! Lecture content saved to {output_filename}")

if __name__ == "__main__":
    main()