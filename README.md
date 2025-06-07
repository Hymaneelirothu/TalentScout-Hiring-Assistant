# TalentScout Hiring Assistant

An intelligent chatbot designed to assist in the initial screening of candidates for TalentScout, a fictional recruitment agency specializing in technology placements.

## Features

- Interactive chat interface for candidate screening
- Information gathering for essential candidate details
- Tech stack-based technical question generation
- Context-aware conversation handling
- Graceful conversation exit mechanism
- Uses local DeepSeek Coder model for responses

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd talentscout-hiring-assistant
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv\Scripts\activate  
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
streamlit run app_streamlit.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Interact with the chatbot by:
   - Providing your personal information
   - Specifying your tech stack
   - Answering technical questions
   - Type 'exit' to end the conversation

## Technical Details

- Built with Streamlit for the user interface
- Uses DeepSeek Coder 6.7B model for natural language processing
- Implements session state management for conversation history
- Handles sensitive information securely
- Runs locally without requiring external API keys

## Model Information

The application uses the DeepSeek Coder 6.7B model, which is:
- Open source and free to use
- Optimized for code and technical conversations
- Runs locally on your machine
- Requires approximately 16GB of RAM for optimal performance

## Prompt Design

The chatbot uses carefully crafted prompts to:
- Gather candidate information systematically
- Generate relevant technical questions based on the declared tech stack
- Maintain conversation context and flow
- Handle unexpected inputs gracefully

## Data Privacy

- All candidate information is handled in compliance with data privacy standards
- No data is permanently stored
- Session data is cleared when the conversation ends
- All processing is done locally on your machine 