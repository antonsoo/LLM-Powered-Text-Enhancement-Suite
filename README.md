# LLM-Powered Text Enhancement Suite

This project is a Streamlit web application that leverages the power of Large Language Models (LLMs) like GPT-3.5-turbo and GPT-4 to provide a suite of text enhancement functionalities. It's designed to be a versatile tool for anyone who works with text, including marketers, content creators, customer service representatives, and developers.

## Features

The app offers the following core features:

*   **Smart Text Completion & Autocorrect:**  Predicts and suggests completions as you type, helping you write faster and more accurately.
*   **Tone Adjustment:**  Modify the tone of your text along multiple dimensions (e.g., formal, informal, humorous, persuasive, empathetic) with fine-grained control over the intensity. You can also specify the intended audience and formality level for more tailored results.
*   **Content Summarization:** Generate concise summaries of longer texts, with the option to focus on specific keywords and control the summary length.
*   **Grammar and Style Improvement:** Enhance the grammar, style, and clarity of your writing.
*   **Model/Prompt Comparison:** Experiment with different LLMs (GPT-3.5-turbo and GPT-4) and various prompts to compare their performance on specific tasks.
*   **Prompt Engineering & Evaluation:** Includes a dedicated section for evaluating and refining prompts. You can edit prompts, test them, and compare results. It also supports saving prompts within a session.
*   **Use Cases:** Provides example use cases in marketing, customer service, content creation, and code documentation to illustrate the practical applications of the app.
*   **File Upload/Download:**  Allows users to upload text files for processing and download the enhanced text.
*   **Clear Input Button:** Added to each tab.

## Getting Started

### Prerequisites

*   Python 3.7 or higher
*   `pip` package manager

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/antonsoo/LLM-Powered-Text-Enhancement-Suite
    cd LLM-Powered-Text-Enhancement-Suite
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Set your OpenAI API Key:**
    *   Create a file named `.streamlit/secrets.toml` in the project directory.
    *   Add your OpenAI API key to the `secrets.toml` file:
        ```toml
        OPENAI_API_KEY = "your_openai_api_key_here"
        ```

### Running the App

1. **Start the Streamlit app:**

    ```bash
    streamlit run app.py
    ```

2. **Access the app:**
    The app will open in your default web browser, usually at `http://localhost:8501`.

    If running from Google Colab, an `ngrok` tunnel will be created. You can get the link to the app from the output of the cell that runs the app.

## Use Cases

This app is designed to be helpful in a variety of real-world scenarios:

*   **Marketing:**
    *   Craft compelling marketing copy tailored to specific target audiences.
    *   Adjust the tone of social media posts, emails, and ad copy.
*   **Customer Service:**
    *   Generate empathetic and professional responses to customer inquiries.
    *   Improve the clarity and tone of customer support emails.
*   **Content Creation:**
    *   Summarize lengthy articles, reports, or research papers.
    *   Generate different variations of content with different tones and styles.
*   **Code Documentation:**
    *   Improve the grammar, style, and clarity of code comments and documentation.
*   **General Writing:**
    *   Enhance the overall quality of any written text.
    *   Get help with writer's block by using the text completion feature.

## Prompt Engineering

The app includes a dedicated section for prompt engineering and evaluation. You can:

*   **Test different prompts:** Experiment with various prompt formulations to see how they affect the results.
*   **Edit prompts:** Modify existing prompts or create your own.
*   **Compare prompts:** Run different prompts side-by-side to compare their performance.
*   **Save prompts:** Temporarily save edited prompts within the app's session.

## Model Comparison

The app allows you to compare the performance of different LLMs (currently GPT-3.5-turbo and GPT-4) on various tasks. This can help you choose the best model for your specific needs.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Anton Soloviev - https://www.upwork.com/freelancers/~01b9d171164a005062
