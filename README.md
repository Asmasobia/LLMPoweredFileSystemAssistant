# LLM Powered File System Assistant

An AI-powered file system assistant that uses LLM tool calling to manage and analyze resume files.

## Features
- **Read Files**: Extract text from PDF, TXT, and DOCX resume files
- **List Files**: Browse directories with optional extension filtering
- **Write Files**: Create new files with auto directory creation
- **Search Files**: Case-insensitive keyword search with context

## Setup

### 1. Clone the project
```bash
git clone https://github.com/yourusername/LLMPoweredFileSystemAssistant.git
cd LLMPoweredFileSystemAssistant

2. Install dependencies
pip install -r requirements.txt

3. Configure API Key
Create a .env file in the project root.
Get a free API key at console.groq.com/keys

4. Run the assistant
python llm_file_assistant.py

Usage Examples
You: List all files in the resumes folder
You: Read the file resumes/resume_john_doe.txt
You: Search for Python in resumes/resume_jane_smith.txt
You: Find resumes mentioning JavaScript experience
You: Create a summary file for resumes/resume_john_doe.txt
You: quit