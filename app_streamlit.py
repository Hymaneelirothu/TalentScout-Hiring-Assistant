import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv
import os
import json
import re
from datetime import datetime
import tempfile

# Load environment variables
load_dotenv()

# Set cache directory to user's temp directory
CACHE_DIR = os.path.join(tempfile.gettempdir(), 'talentscout_cache')
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "candidate_info" not in st.session_state:
    st.session_state.candidate_info = {
        "full_name": "",
        "email": "",
        "phone": "",
        "years_experience": "",
        "desired_position": "",
        "location": "",
        "tech_stack": [],
        "technical_questions": [],
        "conversation_start_time": datetime.now().isoformat()
    }
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "greeting"
if "collected_info" not in st.session_state:
    st.session_state.collected_info = set()
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0

# Initial greeting message
INITIAL_GREETING = """Hello! I'm the TalentScout Hiring Assistant. I'll help you through our initial screening process.

Please provide your information in the following format:
Full Name, Email, Phone Number, Years of Experience, Desired Position, Location, Tech Stack

For example:
John Doe, john@email.com, 1234567890, 3, Software Engineer, New York, Python JavaScript React

You can type 'exit' at any time to end the conversation."""

# Sample technical questions for when model fails to load
SAMPLE_QUESTIONS = {
    "python": [
        "Explain the difference between lists and tuples in Python.",
        "How does Python handle memory management?",
        "What are decorators in Python and how do you use them?"
    ],
    "java": [
        "What is the difference between HashMap and HashTable in Java?",
        "Explain the concept of garbage collection in Java.",
        "What are the main differences between abstract classes and interfaces?"
    ],
    "javascript": [
        "Explain the concept of closures in JavaScript.",
        "What is the difference between let, const, and var?",
        "How does the event loop work in JavaScript?"
    ],
    "react": [
        "What is the difference between state and props in React?",
        "Explain the concept of virtual DOM in React.",
        "What are React hooks and how do you use them?"
    ],
    "node.js": [
        "What is the event-driven architecture in Node.js?",
        "Explain the concept of streams in Node.js.",
        "How does error handling work in Node.js?"
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "Explain the concept of database normalization.",
        "What are indexes and when should you use them?"
    ],
    "mongodb": [
        "What is the difference between MongoDB and traditional SQL databases?",
        "Explain the concept of sharding in MongoDB.",
        "How do you handle relationships in MongoDB?"
    ],
    "docker": [
        "What is the difference between Docker images and containers?",
        "Explain the concept of Docker volumes.",
        "How do you handle environment variables in Docker?"
    ],
    "kubernetes": [
        "What is the difference between a Pod and a Deployment?",
        "Explain the concept of services in Kubernetes.",
        "How do you handle configuration management in Kubernetes?"
    ],
    "aws": [
        "What is the difference between EC2 and Lambda?",
        "Explain the concept of auto-scaling in AWS.",
        "How do you handle security in AWS?"
    ],
    "ml": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of overfitting and how to prevent it.",
        "What are the main types of machine learning algorithms?"
    ],
    "dl": [
        "What are the different types of neural network architectures?",
        "Explain the concept of backpropagation.",
        "What are activation functions and why are they important?"
    ],
    "tensorflow": [
        "What is the difference between TensorFlow 1.x and 2.x?",
        "Explain the concept of eager execution in TensorFlow.",
        "How do you handle model deployment in TensorFlow?"
    ],
    "pytorch": [
        "What is the difference between PyTorch and TensorFlow?",
        "Explain the concept of dynamic computation graphs in PyTorch.",
        "How do you handle model serialization in PyTorch?"
    ],
    "git": [
        "What is the difference between git merge and git rebase?",
        "Explain the concept of branching strategies.",
        "How do you handle merge conflicts?"
    ],
    "jenkins": [
        "What is the difference between Jenkins pipeline and freestyle jobs?",
        "Explain the concept of Jenkins agents.",
        "How do you handle environment variables in Jenkins?"
    ],
    "linux": [
        "What is the difference between process and thread in Linux?",
        "Explain the concept of file permissions in Linux.",
        "How do you handle process management in Linux?"
    ],
    "networking": [
        "What is the difference between TCP and UDP?",
        "Explain the concept of DNS resolution.",
        "How do you handle network security?"
    ],
    "security": [
        "What is the difference between authentication and authorization?",
        "Explain the concept of OAuth 2.0.",
        "How do you handle secure password storage?"
    ],
    "devops": [
        "What is the difference between CI and CD?",
        "Explain the concept of infrastructure as code.",
        "How do you handle monitoring and logging?"
    ]
}

def get_sample_questions(tech_stack):
    questions = []
    for tech in tech_stack:
        tech_lower = tech.lower()
        # Try to find the technology in our sample questions
        found = False
        for key in SAMPLE_QUESTIONS.keys():
            if tech_lower in key or key in tech_lower:
                questions.extend([f"{tech}: {q}" for q in SAMPLE_QUESTIONS[key]])
                found = True
                break
        if not found:
            # If technology not found, add a generic message
            questions.append(f"{tech}: I don't have specific questions for this technology. Please provide your experience with {tech}.")
    return questions

def validate_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_phone(phone):
    # Remove any spaces or special characters
    phone = re.sub(r'[^\d+]', '', phone)
    return len(phone) >= 10

def parse_user_input(user_input):
    # Split the input by commas and clean up each part
    parts = [part.strip() for part in user_input.split(',')]
    
    # Ensure we have enough parts
    if len(parts) < 7:
        return None, "Please provide all required information in the format: Name, Email, Phone, Experience, Position, Location, Tech Stack"
    
    # Extract and validate information
    name = parts[0]
    email = parts[1]
    phone = parts[2]
    experience = parts[3]
    position = parts[4]
    location = parts[5]
    tech_stack = [tech.strip() for tech in parts[6].split()]
    
    # Validate email
    if not validate_email(email):
        return None, "Please provide a valid email address (e.g., user@example.com)"
    
    # Validate phone
    if not validate_phone(phone):
        return None, "Please provide a valid phone number (at least 10 digits)"
    
    return {
        "full_name": name,
        "email": email,
        "phone": phone,
        "years_experience": experience,
        "desired_position": position,
        "location": location,
        "tech_stack": tech_stack
    }, None

def extract_tech_stack(text):
    # Common tech stack keywords
    tech_keywords = {
        'languages': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin'],
        'frameworks': ['django', 'flask', 'spring', 'react', 'angular', 'vue', 'express', 'rails', 'laravel', 'asp.net'],
        'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'oracle', 'sql server'],
        'tools': ['docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git', 'jira', 'ansible', 'terraform'],
        'ai_ml': ['ml', 'machine learning', 'deep learning', 'dl', 'ai', 'artificial intelligence', 'tensorflow', 'pytorch', 'scikit-learn']
    }
    
    found_tech = []
    text_lower = text.lower()
    
    for category, keywords in tech_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_tech.append(keyword)
    
    return found_tech

def generate_technical_questions(tech_stack):
    questions = []
    for tech in tech_stack:
        # Create a detailed prompt for question generation
        prompt = f"""As a technical interviewer, generate 3 relevant technical questions about {tech} that would be appropriate for a job interview.
        The questions should:
        1. Be specific to {tech} and its practical applications
        2. Vary in difficulty (basic, intermediate, advanced)
        3. Focus on real-world scenarios and problem-solving
        4. Be relevant to the position of {st.session_state.candidate_info['desired_position']}
        5. Consider the candidate's {st.session_state.candidate_info['years_experience']} years of experience

        Format each question on a new line starting with a number (1., 2., 3.)"""
        
        try:
            # Try to get response from the model
            response = get_ai_response([{"role": "system", "content": prompt}])
            
            # Process the response to extract questions
            question_lines = [line.strip() for line in response.split('\n') if line.strip()]
            tech_questions = []
            
            for line in question_lines:
                # Remove numbering if present
                question = re.sub(r'^\d+\.\s*', '', line)
                if question:
                    tech_questions.append(question)
            
            # If we got valid questions, add them
            if tech_questions:
                questions.extend([f"{tech}: {q}" for q in tech_questions[:3]])
            else:
                # Fallback if no valid questions were generated
                questions.append(f"{tech}: Please describe your experience with {tech} and any significant projects you've worked on.")
        
        except Exception as e:
            # Fallback in case of any error
            questions.append(f"{tech}: Please describe your experience with {tech} and any significant projects you've worked on.")
    
    return questions

@st.cache_resource
def load_model():
    try:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        model_name = "deepseek-ai/deepseek-coder-6.7b-base"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=CACHE_DIR,
            local_files_only=False
        )
        st.session_state.model_loaded = True
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False
        return None, None

def get_ai_response(messages):
    try:
        if not st.session_state.model_loaded:
            # If model isn't loaded, use a simpler prompt for question generation
            tech = messages[0]["content"].split("about ")[1].split(" that")[0]
            return f"""1. What is your experience with {tech} and how have you used it in your projects?
2. Can you explain a challenging problem you solved using {tech}?
3. How do you stay updated with the latest developments in {tech}?"""
        
        model, tokenizer = load_model()
        if model is None or tokenizer is None:
            # If model loading failed, use the same fallback
            tech = messages[0]["content"].split("about ")[1].split(" that")[0]
            return f"""1. What is your experience with {tech} and how have you used it in your projects?
2. Can you explain a challenging problem you solved using {tech}?
3. How do you stay updated with the latest developments in {tech}?"""
        
        conversation = ""
        for msg in messages:
            role = "Assistant" if msg["role"] == "assistant" else "Human" if msg["role"] == "user" else "System"
            conversation += f"{role}: {msg['content']}\n"
        conversation += "Assistant: "
        
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_length=500,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant: ")[-1].strip()
        return response
    except Exception as e:
        # Fallback in case of any error
        tech = messages[0]["content"].split("about ")[1].split(" that")[0]
        return f"""1. What is your experience with {tech} and how have you used it in your projects?
2. Can you explain a challenging problem you solved using {tech}?
3. How do you stay updated with the latest developments in {tech}?"""

def process_user_input(user_input):
    if user_input.lower() == "exit":
        return "Thank you for your time! We'll review your information and get back to you soon."
    
    # Handle technical interview stage
    if st.session_state.current_stage == "tech_stack_analysis":
        questions = st.session_state.candidate_info["technical_questions"]
        if st.session_state.current_question_index < len(questions):
            # Evaluate the answer
            current_question = questions[st.session_state.current_question_index]
            tech = current_question.split(":")[0].strip()
            
            # Move to next question
            st.session_state.current_question_index += 1
            
            if st.session_state.current_question_index < len(questions):
                next_question = questions[st.session_state.current_question_index]
                return f"Thank you for your answer. Let's move on to the next question:\n{next_question}"
            else:
                st.session_state.current_stage = "conclusion"
                return "Thank you for completing the technical interview! We'll review your responses and get back to you soon."
    
    # Try to parse the input as a complete profile
    parsed_info, error = parse_user_input(user_input)
    if error:
        return error
    
    if parsed_info:
        # Update candidate information
        st.session_state.candidate_info.update(parsed_info)
        st.session_state.current_stage = "tech_stack_analysis"
        st.session_state.current_question_index = 0
        
        # Generate technical questions
        questions = generate_technical_questions(parsed_info["tech_stack"])
        st.session_state.candidate_info["technical_questions"] = questions
        
        # Return confirmation and first technical question
        response = f"""Thank you for providing your information! I've recorded:
- Name: {parsed_info['full_name']}
- Position: {parsed_info['desired_position']}
- Tech Stack: {', '.join(parsed_info['tech_stack'])}

Let's proceed with some technical questions. Here's the first one:
{questions[0] if questions else 'No technical questions generated.'}"""
        return response
    
    return get_ai_response([{"role": "user", "content": user_input}])

def main():
    st.title("TalentScout Hiring Assistant")
    
    # Display initial greeting if no messages
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": INITIAL_GREETING
        })
    
    # Sidebar for collected information
    with st.sidebar:
        st.header("Collected Information")
        for key, value in st.session_state.candidate_info.items():
            if key not in ["technical_questions", "conversation_start_time"]:
                st.write(f"{key.replace('_', ' ').title()}: {value}")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process user input and get response
        response = process_user_input(prompt)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main() 