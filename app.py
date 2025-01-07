import streamlit as st
import openai
from functools import lru_cache
import time
import random
import os

# In Hugging Face Spaces, we can retrieve the API key from st.secrets.
# Make sure you've set an "OPENAI_API_KEY" and "NGROK_AUTH_TOKEN" secrets under your Space settings.
NGROK_AUTH_TOKEN = st.secrets["NGROK_AUTH_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# --- Configuration ---
OPENAI_MODEL = "gpt-3.5-turbo"  # You can make this configurable in the UI

# --- Prompt Definitions ---
tone_prompts = {
    "Rewrite the following text to be more {tone_dimensions} with intensity {tone_strength}:\n\n{text}": True,
    "Adjust the tone of the following text to reflect {tone_dimensions} at intensity level {tone_strength}:\n\n{text}": True,
    "Modify the text below to have a tone that is {tone_dimensions}, with a strength of {tone_strength}:\n\n{text}": True
}

summarization_prompts = {
    "Summarize the following text {length_instruction}, with a focus on these keywords: {keyword_str}.\n\n{text}": True,
    "Provide a summary of the following text {length_instruction}, emphasizing the keywords: {keyword_str}.\n\n{text}": True,
    "Condense the following text into a summary {length_instruction}, while highlighting these keywords: {keyword_str}.\n\n{text}": True
}

grammar_style_prompts = {
    "Please improve the grammar and style of the following text, correct any errors:\n\n{text}": True,
    "Review the following text for grammar and style, and provide corrections:\n\n{text}": True,
    "Edit the following text to enhance its grammar and style:\n\n{text}": True
}

# --- Helper Functions ---
@lru_cache(maxsize=128)
def get_completion_with_retry(prompt, model=OPENAI_MODEL, max_tokens=50, temperature=0.7, n=1, stop=None, max_retries=3, delay=1):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                # n=n,  # 'n' parameter is deprecated for chat-based models
                stop=stop,
            )
            # Adapt to the new response format (for chat-based models)
            return [choice.message.content for choice in response.choices]
        except openai.OpenAIError as e:
            st.error(f"OpenAI API Error: {e}")
            if "rate limit" in str(e).lower():
                wait_time = delay * (2 ** retries) + random.uniform(0, delay)
                st.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                st.error(f"Non-rate limit error: {e}. Retrying...")
                time.sleep(delay)
                retries += 1
    st.error(f"Failed to get completion after {max_retries} retries.")
    return []

def adjust_tone(text, tone_dimensions, tone_strength, audience, formality, prompt_index=0, model=OPENAI_MODEL):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        # Incorporate metadata into the prompt
        prompt = list(tone_prompts.keys())[prompt_index].format(
            tone_dimensions=tone_dimensions,
            tone_strength=tone_strength,
            audience=audience,  # Add audience to the prompt
            formality=formality,  # Add formality to the prompt
            text=text
        )
        modified_prompt = f"The intended audience for this text is: {audience}. The desired level of formality is: {formality}.\n\n{prompt}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can adjust the tone of text."},
                {"role": "user", "content": modified_prompt}  # Use the modified prompt
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        if "rate limit" in str(e).lower():
            st.error("The API rate limit has been exceeded. Please try again later.")
        elif "invalid credentials" in str(e).lower():
            st.error("Invalid OpenAI API key. Please check your API key and try again.")
        else:
            st.error("An unexpected error occurred while communicating with the OpenAI API.")
        return None
    except IndexError:
        st.error(f"Invalid prompt index: {prompt_index}. Using default prompt.")
        prompt = list(tone_prompts.keys())[0].format(
            tone_dimensions=tone_dimensions,
            tone_strength=tone_strength,
            audience=audience,
            formality=formality,
            text=text
        )
        modified_prompt = f"The intended audience for this text is: {audience}. The desired level of formality is: {formality}.\n\n{prompt}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can adjust the tone of text."},
                {"role": "user", "content": modified_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

def summarize_text(text, keywords, summary_length, prompt_index=0, model=OPENAI_MODEL):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    progress_bar = st.progress(0)  # Initialize progress bar
    try:
        length_instruction = {
            "short": "in 1-2 sentences",
            "medium": "in 3-4 sentences",
            "long": "in 5-7 sentences"
        }[summary_length]
        progress_bar.progress(25)

        keyword_str = ", ".join(keywords) if keywords else ""
        prompt = list(summarization_prompts.keys())[prompt_index].format(length_instruction=length_instruction, keyword_str=keyword_str, text=text)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can summarize text."},
                {"role": "user", "content": prompt}
            ]
        )
        progress_bar.progress(75)
        summarized_text = response.choices[0].message.content.strip()
        progress_bar.progress(100)
        return summarized_text
    except openai.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        if "rate limit" in str(e).lower():
            st.error("The API rate limit has been exceeded. Please try again later.")
        elif "invalid credentials" in str(e).lower():
            st.error("Invalid OpenAI API key. Please check your API key and try again.")
        else:
            st.error("An unexpected error occurred while communicating with the OpenAI API.")
        return None
    except (KeyError, IndexError) as e:
        st.error(f"Error: {e}. Using default prompt and length instruction.")
        prompt = list(summarization_prompts.keys())[0].format(length_instruction="in 1-2 sentences", keyword_str="", text=text)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can summarize text."},
                {"role": "user", "content": prompt}
            ]
        )
        progress_bar.progress(100)
        return response.choices[0].message.content.strip()
    finally:
        progress_bar.empty()  # Remove the progress bar when done

def improve_grammar_and_style(text, focus="both", prompt_index=0, model=OPENAI_MODEL):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        if focus == "both":
            prompt = list(grammar_style_prompts.keys())[prompt_index].format(text=text)
        elif focus == "grammar":
            prompt = f"Please correct any grammatical errors in the following text:\n\n{text}"
        elif focus == "style":
            prompt = f"Please improve the writing style of the following text:\n\n{text}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can improve grammar and style of text."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        if "rate limit" in str(e).lower():
            st.error("The API rate limit has been exceeded. Please try again later.")
        elif "invalid credentials" in str(e).lower():
            st.error("Invalid OpenAI API key. Please check your API key and try again.")
        else:
            st.error("An unexpected error occurred while communicating with the OpenAI API.")
        return None
    except IndexError:
        st.error(f"Invalid prompt index: {prompt_index}. Using default prompt.")
        prompt = list(grammar_style_prompts.keys())[0].format(text=text)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that can improve grammar and style of text."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

# --- Streamlit App ---

def main():
    st.title("LLM-Powered Text Enhancement Suite")

    # --- Sidebar ---
    st.sidebar.header("Settings")
    selected_model = st.sidebar.selectbox("Choose a model:", ["gpt-3.5-turbo", "gpt-4"])

    # --- Main Content Area ---
    with st.expander("Instructions"):
        st.write("""
            This app allows you to enhance text using various LLM-powered functions. 
            Select a model in the sidebar and then choose a function from the tabs below.
            You can also select different prompts for each function to experiment with the results.
        """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Completion", "Tone", "Summarize", "Grammar", "Comparison"])

    # --- Use Cases Section ---
    with st.expander("Use Cases and Examples"):
        st.write("""
            This app can be used in various real-world scenarios, such as:

            **Marketing:**
            - Automatically adjust the tone of marketing copy to target different customer segments.
            - Example:
                - Input Text: "Our new product is revolutionary!"
                - Tone Dimensions: "Excited, persuasive"
                - Audience: "Young adults"
                - Result: "Get ready to be blown away by our groundbreaking new product! It's a game-changer!"

            **Customer Service:**
            - Generate empathetic and helpful responses to customer inquiries.
            - Example:
                - Input Text: "I'm having trouble with my order."
                - Tone Dimensions: "Empathetic, understanding"
                - Audience: "Frustrated customer"
                - Result: "I'm so sorry you're having trouble with your order. I understand how frustrating that can be. Let me help you resolve this issue."

            **Content Creation:**
            - Summarize lengthy articles or reports into concise summaries for busy executives.
            - Example:
                - Input Text: (A long article about a new scientific discovery)
                - Keywords: "scientific discovery, impact, future research"
                - Summary Length: "short"
                - Result: (A 1-2 sentence summary highlighting the key findings and implications of the discovery)

            **Code Documentation:**
            - Use the grammar and style improvement feature to automatically enhance code documentation.
            - Example:
                - Input Text: "This function takes a list of strings as input and returns a dictionary containing the frequency of each string in the list."
                - Improvement Focus: "Style"
                - Result: "This function accepts a list of strings and outputs a dictionary that maps each string to its frequency within the list."
        """)

        # Example data for each use case
        use_case_examples = {
            "Marketing": {
                "input_text": "Our new product is revolutionary!",
                "tone_dimensions": ["excited", "persuasive"],
                "tone_strength": 0.8,
                "audience": "Young adults",
                "formality": "informal",
            },
            "Customer Service": {
                "input_text": "I'm having trouble with my order.",
                "tone_dimensions": ["empathetic", "understanding"],
                "tone_strength": 0.9,
                "audience": "Frustrated customer",
                "formality": "neutral",
            },
            "Content Creation": {
                "input_text": "Paste a lengthy article here...",  # Placeholder for a long article
                "keywords": "scientific discovery, impact, future research",
                "summary_length": "short",
            },
            "Code Documentation": {
                "input_text": "This function takes a list of strings as input and returns a dictionary containing the frequency of each string in the list.",
                "focus": "style"
            }
        }

        # Dropdown to select a use case
        selected_use_case = st.selectbox("Select a Use Case:", list(use_case_examples.keys()), key="use_case_select")

        # Button to load the example data
        if st.button("Load Example", key="load_example"):
            example_data = use_case_examples[selected_use_case]

            # Update the relevant input fields based on the selected use case
            if selected_use_case in ["Marketing", "Customer Service"]:
                st.session_state.tone_input = example_data["input_text"]
                st.session_state.tone_dimensions = example_data["tone_dimensions"]
                st.session_state.tone_strength = example_data["tone_strength"]
                st.session_state.audience = example_data["audience"]
                st.session_state.formality = example_data["formality"]
            elif selected_use_case == "Content Creation":
                st.session_state.summary_input = example_data["input_text"]
                st.session_state.summary_keywords = example_data["keywords"]
                st.session_state.summary_length = example_data["summary_length"]
            elif selected_use_case == "Code Documentation":
                st.session_state.grammar_input = example_data["input_text"]
                st.session_state.grammar_focus = example_data["focus"]
        # END OF USE-CASE SECTION

    with tab1:
        st.header("Smart Text Completion & Autocorrect")
        user_input = st.text_area("Enter text here:", key="completion_input")

        # File Upload for Completion
        uploaded_file_completion = st.file_uploader("Upload a text file for completion:", type=["txt"], key="file_uploader_completion")
        if uploaded_file_completion is not None:
            # Read the file content
            file_content = uploaded_file_completion.read().decode("utf-8")
            # Update the input field with the file content
            st.session_state.completion_input = file_content
            user_input = file_content

        # Clear button for Completion
        if st.button("Clear Input", key="clear_completion_tab1"):
            st.session_state.completion_input = ""
            user_input = ""

        if user_input:
            if st.button("Generate Completions", key="completion_button"):
                with st.spinner("Generating completions..."):
                    completions = get_completion_with_retry(user_input, model=selected_model)
                if completions:
                    st.subheader("Completions:")
                    for i, choice in enumerate(completions):
                        st.write(f"{i+1}. {choice}")

                    # Download for Completion
                    st.download_button(
                        label="Download Completions",
                        data="\n".join(completions).encode("utf-8"),
                        file_name="completions.txt",
                        mime="text/plain",
                        key="download_completion"
                    )
                else:
                    st.write("No completions found.")

    with tab2:
        st.header("Tone Adjustment")
        tone_input = st.text_area("Enter text to adjust tone:", key="tone_input")

        # File Upload for Tone Adjustment
        uploaded_file_tone = st.file_uploader("Upload a text file for tone adjustment:", type=["txt"], key="file_uploader_tone")
        if uploaded_file_tone is not None:
            # Read the file content
            file_content = uploaded_file_tone.read().decode("utf-8")
            # Update the input field with the file content
            st.session_state.tone_input = file_content
            tone_input = file_content
        
        # Clear button for Tone Adjustment
        if st.button("Clear Input", key="clear_tone_tab2"):
            st.session_state.tone_input = ""
            tone_input = ""
        
        tone_dimensions = st.multiselect("Select tone dimensions:", ["formal", "informal", "humorous", "persuasive", "empathetic", "assertive", "conciliatory", "polite"], default=["formal", "polite"], key="tone_dimensions")
        tone_strength = st.slider("Tone Strength", 0.0, 1.0, 0.5, key="tone_strength")
        # Add input fields for metadata
        audience = st.text_input("Intended Audience:", key="audience")
        formality = st.selectbox("Formality Level:", ["very informal", "informal", "neutral", "formal", "very formal"], key="formality")

        # Prompt selection for Tone Adjustment
        selected_tone_prompt_index = st.selectbox("Tone Adjustment Prompt:", range(len(tone_prompts)), format_func=lambda x: f"Prompt {x+1}", key="tone_prompt_select")

        # Convert list of dimensions to a comma-separated string
        tone_dimensions_str = ", ".join(tone_dimensions)

        if st.button("Adjust Tone", key="tone_button"):
            with st.spinner("Adjusting tone..."):
                # Pass metadata to the adjust_tone function
                adjusted_text = adjust_tone(tone_input, tone_dimensions_str, tone_strength, audience, formality, prompt_index=selected_tone_prompt_index, model=selected_model)
            if adjusted_text:
                st.subheader("Adjusted Text:")
                st.write(adjusted_text)

                # Download for Tone Adjustment
                st.download_button(
                    label="Download Adjusted Text",
                    data=adjusted_text.encode("utf-8"),
                    file_name="adjusted_text.txt",
                    mime="text/plain",
                    key="download_tone"
                )
            else:
                st.write("Tone adjustment failed.")

            # --- Prompt Evaluation Section ---
            st.header("Prompt Evaluation")
            st.write("Here, you can compare the results of different prompts for each function.")
            selected_function = st.selectbox("Select a function to evaluate:", ["Tone Adjustment", "Summarization", "Grammar and Style Improvement"], key="function_select")

            if selected_function == "Tone Adjustment":
                st.subheader("Tone Adjustment Prompts")
                
                # Create a dictionary to store the edited prompts
                edited_tone_prompts = st.session_state.get("edited_tone_prompts", tone_prompts.copy())

                # Create a list to store the keys of edited_tone_prompts
                edited_tone_prompts_keys = list(edited_tone_prompts.keys())

                for i, prompt in enumerate(edited_tone_prompts_keys):
                    st.write(f"**Prompt {i+1}:**")
                    
                    # Use text_area for editable prompts, and provide unique keys
                    edited_prompt = st.text_area(f"Edit Prompt {i+1}:", prompt, key=f"tone_prompt_edit_{i}")

                    # Update the dictionary with the edited prompt
                    edited_tone_prompts[edited_prompt] = edited_tone_prompts.pop(prompt)

                    if st.button(f"Test Prompt {i+1}", key=f"tone_test_{i}"):
                        with st.spinner("Testing prompt..."):
                            # Use the same input text and parameters for fair comparison
                            adjusted_text = adjust_tone(tone_input, tone_dimensions_str, tone_strength, audience, formality, prompt_index=i, model=selected_model)
                        if adjusted_text:
                            st.subheader("Result:")
                            st.write(adjusted_text)
                        else:
                            st.write("Prompt test failed.")

                # Save the edited prompts to session state
                st.session_state.edited_tone_prompts = edited_tone_prompts

            elif selected_function == "Summarization":
              st.subheader("Summarization Prompts")

              # Create a dictionary to store the edited prompts
              edited_summarization_prompts = st.session_state.get("edited_summarization_prompts", summarization_prompts.copy())

              # Create a list to store the keys of edited_summarization_prompts
              edited_summarization_prompts_keys = list(edited_summarization_prompts.keys())

              for i, prompt in enumerate(edited_summarization_prompts_keys):
                  st.write(f"**Prompt {i+1}:**")
                  
                  # Use text_area for editable prompts, and provide unique keys
                  edited_prompt = st.text_area(f"Edit Prompt {i+1}:", prompt, key=f"summarization_prompt_edit_{i}")

                  # Update the dictionary with the edited prompt
                  edited_summarization_prompts[edited_prompt] = edited_summarization_prompts.pop(prompt)

                  if st.button(f"Test Prompt {i+1}", key=f"summarization_test_{i}"):
                      with st.spinner("Testing prompt..."):
                        keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []
                        # Use the same input text and parameters for fair comparison
                        summarized_text = summarize_text(summary_input, keywords_list, summary_length, prompt_index=i, model=selected_model)
                      if summarized_text:
                          st.subheader("Result:")
                          st.write(summarized_text)
                      else:
                          st.write("Prompt test failed.")

              # Save the edited prompts to session state
              st.session_state.edited_summarization_prompts = edited_summarization_prompts

            elif selected_function == "Grammar and Style Improvement":
              st.subheader("Grammar and Style Improvement Prompts")

              # Create a dictionary to store the edited prompts
              edited_grammar_style_prompts = st.session_state.get("edited_grammar_style_prompts", grammar_style_prompts.copy())

              # Create a list to store the keys of edited_grammar_style_prompts
              edited_grammar_style_prompts_keys = list(edited_grammar_style_prompts.keys())

              for i, prompt in enumerate(edited_grammar_style_prompts_keys):
                  st.write(f"**Prompt {i+1}:**")

                  # Use text_area for editable prompts, and provide unique keys
                  edited_prompt = st.text_area(f"Edit Prompt {i+1}:", prompt, key=f"grammar_style_prompt_edit_{i}")

                  # Update the dictionary with the edited prompt
                  edited_grammar_style_prompts[edited_prompt] = edited_grammar_style_prompts.pop(prompt)

                  if st.button(f"Test Prompt {i+1}", key=f"grammar_style_test_{i}"):
                      with st.spinner("Testing prompt..."):
                        # Use the same input text and parameters for fair comparison
                        improved_text = improve_grammar_and_style(grammar_input, improvement_focus, prompt_index=i, model=selected_model)
                      if improved_text:
                          st.subheader("Result:")
                          st.write(improved_text)
                      else:
                          st.write("Prompt test failed.")

              # Save the edited prompts to session state
              st.session_state.edited_grammar_style_prompts = edited_grammar_style_prompts
            
            # Save the edited prompts to session state (for all functions)
            if st.button("Save Prompts"):
                st.success("Prompts saved temporarily (within this session).")

    with tab3:
        st.header("Content Summarization with Keyword Focus")
        summary_input = st.text_area("Enter text to summarize:", key="summary_input")

        # File Upload for Summarization
        uploaded_file_summary = st.file_uploader("Upload a text file for summarization:", type=["txt"], key="file_uploader_summary")
        if uploaded_file_summary is not None:
            # Read the file content
            file_content = uploaded_file_summary.read().decode("utf-8")
            # Update the input field with the file content
            st.session_state.summary_input = file_content
            summary_input = file_content

        # Clear button for Summarization
        if st.button("Clear Input", key="clear_summary_tab3"):
            st.session_state.summary_input = ""
            summary_input = ""
        
        keywords = st.text_input("Enter keywords (comma-separated):", key="summary_keywords")
        summary_length = st.selectbox("Summary Length:", ["short", "medium", "long"], key="summary_length")

        # Prompt selection for Summarization
        selected_summary_prompt_index = st.selectbox("Summarization Prompt:", range(len(summarization_prompts)), format_func=lambda x: f"Prompt {x+1}", key="summary_prompt_select")

        if st.button("Summarize", key="summary_button"):
            with st.spinner("Summarizing..."):
                keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []
                summarized_text = summarize_text(summary_input, keywords_list, summary_length, prompt_index=selected_summary_prompt_index, model=selected_model)
            if summarized_text:
                st.subheader("Summarized Text:")
                st.write(summarized_text)

                # Download for Summarization
                st.download_button(
                    label="Download Summarized Text",
                    data=summarized_text.encode("utf-8"),
                    file_name="summarized_text.txt",
                    mime="text/plain",
                    key="download_summary"
                )
            else:
                st.write("Summarization failed.")

    with tab4:
        st.header("Grammar and Style Improvement")
        grammar_input = st.text_area("Enter text to improve:", key="grammar_input")

        # File Upload for Grammar and Style Improvement
        uploaded_file_grammar = st.file_uploader("Upload a text file for grammar and style improvement:", type=["txt"], key="file_uploader_grammar")
        if uploaded_file_grammar is not None:
            # Read the file content
            file_content = uploaded_file_grammar.read().decode("utf-8")
            # Update the input field with the file content
            st.session_state.grammar_input = file_content
            grammar_input = file_content
        
        # Clear button for Grammar and Style Improvement
        if st.button("Clear Input", key="clear_grammar_tab4"):
            st.session_state.grammar_input = ""
            grammar_input = ""
        
        improvement_focus = st.selectbox("Improvement Focus:", ["both", "grammar", "style"], key="grammar_focus")

        # Prompt selection for Grammar and Style Improvement
        selected_grammar_prompt_index = st.selectbox("Grammar/Style Prompt:", range(len(grammar_style_prompts)), format_func=lambda x: f"Prompt {x+1}", key="grammar_prompt_select")

        if st.button("Improve Text", key="grammar_button"):
            with st.spinner("Improving text..."):
                improved_text = improve_grammar_and_style(grammar_input, improvement_focus, prompt_index=selected_grammar_prompt_index, model=selected_model)
            if improved_text:
                st.subheader("Improved Text:")
                st.write(improved_text)

                # Download for Grammar and Style Improvement
                st.download_button(
                    label="Download Improved Text",
                    data=improved_text.encode("utf-8"),
                    file_name="improved_text.txt",
                    mime="text/plain",
                    key="download_grammar"
                )
            else:
                st.write("Text improvement failed.")

    with tab5:
        st.header("Model/Prompt Comparison")

        # Select function to compare
        selected_function = st.selectbox("Select a function to compare:", ["Tone Adjustment", "Summarization", "Grammar and Style Improvement"], key="comparison_function_select")

        # Input text area (for all functions)
        comparison_input_text = st.text_area("Enter text for comparison:", key="comparison_input")

        # --- Model/Prompt Selection ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model/Prompt 1")
            if selected_function == "Tone Adjustment":
                selected_model_1 = st.selectbox("Model 1:", ["gpt-3.5-turbo", "gpt-4"], key="model_select_1")
                selected_prompt_index_1 = st.selectbox("Prompt 1:", range(len(tone_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_select_1")
            elif selected_function == "Summarization":
                selected_model_1 = st.selectbox("Model 1:", ["gpt-3.5-turbo", "gpt-4"], key="model_summary_select_1")
                selected_prompt_index_1 = st.selectbox("Summarization Prompt 1:", range(len(summarization_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_summary_select_1")
            elif selected_function == "Grammar and Style Improvement":
                selected_model_1 = st.selectbox("Model 1:", ["gpt-3.5-turbo", "gpt-4"], key="model_grammar_select_1")
                selected_prompt_index_1 = st.selectbox("Grammar/Style Prompt 1:", range(len(grammar_style_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_grammar_select_1")

        with col2:
            st.subheader("Model/Prompt 2")
            if selected_function == "Tone Adjustment":
                selected_model_2 = st.selectbox("Model 2:", ["gpt-3.5-turbo", "gpt-4"], key="model_select_2")
                selected_prompt_index_2 = st.selectbox("Prompt 2:", range(len(tone_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_select_2")
            elif selected_function == "Summarization":
                selected_model_2 = st.selectbox("Model 2:", ["gpt-3.5-turbo", "gpt-4"], key="model_summary_select_2")
                selected_prompt_index_2 = st.selectbox("Summarization Prompt 2:", range(len(summarization_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_summary_select_2")
            elif selected_function == "Grammar and Style Improvement":
                selected_model_2 = st.selectbox("Model 2:", ["gpt-3.5-turbo", "gpt-4"], key="model_grammar_select_2")
                selected_prompt_index_2 = st.selectbox("Grammar/Style Prompt 2:", range(len(grammar_style_prompts)), format_func=lambda x: f"Prompt {x+1}", key="prompt_grammar_select_2")

        # --- Run Comparison ---
        if st.button("Compare", key="compare_button"):
            with st.spinner("Running comparison..."):
                if selected_function == "Tone Adjustment":
                    # Get parameters from session state
                    tone_dimensions_str = st.session_state.tone_dimensions_str
                    tone_strength = st.session_state.tone_strength
                    audience = st.session_state.audience
                    formality = st.session_state.formality

                    # Run the function with the selected models/prompts
                    adjusted_text_1 = adjust_tone(comparison_input_text, tone_dimensions_str, tone_strength, audience, formality, selected_prompt_index_1, selected_model_1)
                    adjusted_text_2 = adjust_tone(comparison_input_text, tone_dimensions_str, tone_strength, audience, formality, selected_prompt_index_2, selected_model_2)

                    # Display results side-by-side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Result (Model/Prompt 1)")
                        st.write(adjusted_text_1)
                    with col2:
                        st.subheader("Result (Model/Prompt 2)")
                        st.write(adjusted_text_2)

                elif selected_function == "Summarization":
                   # Get parameters from session state
                    keywords_list = [k.strip() for k in st.session_state.summary_keywords.split(",")] if st.session_state.summary_keywords else []
                    summary_length = st.session_state.summary_length

                    # Run the function with the selected models/prompts
                    summarized_text_1 = summarize_text(comparison_input_text, keywords_list, summary_length, selected_prompt_index_1, selected_model_1)
                    summarized_text_2 = summarize_text(comparison_input_text, keywords_list, summary_length, selected_prompt_index_2, selected_model_2)

                    # Display results side-by-side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Result (Model/Prompt 1)")
                        st.write(summarized_text_1)
                    with col2:
                        st.subheader("Result (Model/Prompt 2)")
                        st.write(summarized_text_2)

                elif selected_function == "Grammar and Style Improvement":
                    # Get parameters from session state
                    improvement_focus = st.session_state.grammar_focus

                    # Run the function with the selected models/prompts
                    improved_text_1 = improve_grammar_and_style(comparison_input_text, improvement_focus, selected_prompt_index_1, selected_model_1)
                    improved_text_2 = improve_grammar_and_style(comparison_input_text, improvement_focus, selected_prompt_index_2, selected_model_2)

                    # Display results side-by-side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Result (Model/Prompt 1)")
                        st.write(improved_text_1)
                    with col2:
                        st.subheader("Result (Model/Prompt 2)")
                        st.write(improved_text_2)

if __name__ == "__main__":
    main()
