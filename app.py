import speech_recognition as sr
import pyttsx3
import threading
import time
import queue
import requests
import json
import os
import numpy as np
from dotenv import load_dotenv


load_dotenv()

class SpeechToSpeechBot:
    def __init__(self, use_rag=False):
        """Initialize the Speech-to-Speech bot with all required components."""
      
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 400
        
        
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0-1)
        
        # Get available voices and set a clear voice
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        
        # Conversation state
        self.conversation_history = []
        self.speaking = False
        self.listening = False
        self.should_stop = False
        self.interrupt_queue = queue.Queue()
        
        # Advanced interrupt settings
        self.interrupt_sensitivity = 1.5  
        self.interrupt_cooldown = 0.2    
        self.last_interrupt_time = 0      
        self.interrupt_energy_history = []  
        
        # API settings for Groq
        self.api_key = os.getenv("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-70b-8192"
        
        # RAG settings
        self.use_rag = use_rag
        if self.use_rag:
            self.initialize_rag()
    
    def initialize_rag(self):
        """Initialize RAG components."""
        try:
            # Placeholder for RAG implementation
            print("RAG system initialized")
            self.rag_enabled = True
        except Exception as e:
            print(f"Failed to initialize RAG: {e}")
            self.rag_enabled = False
    
    def listen(self):
        """Listen for speech input and convert to text."""
        self.listening = True
        with sr.Microphone() as source:
            print("Listening...")
            
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Processing speech...")
                
                # Convert speech to text
                text = self.recognizer.recognize_google(audio)
                print(f"You said: {text}")
                return text
            
            except sr.WaitTimeoutError:
                print("No speech detected")
                return None
            
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None
            
            finally:
                self.listening = False
    
    def process_with_llm(self, text):
        """Process text with Groq LLM and generate response."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": text})
        
        # Check if RAG should be used
        context = ""
        if self.use_rag and self.rag_enabled:
            context = self.retrieve_relevant_information(text)
        
        try:
            # Prepare the payload for the Groq API
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Format messages for the API
            messages = []
            # Add system message with RAG context if available
            if context:
                messages.append({
                    "role": "system", 
                    "content": f"You are a helpful assistant. Use the following information to answer the user's question if relevant: {context}"
                })
            else:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful assistant. Keep responses concise yet informative."
                })
            
            # Add conversation history
            for msg in self.conversation_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
            }
            
            # Call the Groq API
            start_time = time.time()
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload
            )
            api_time = time.time() - start_time
            print(f"Groq API response time: {api_time:.2f} seconds")
            
            if response.status_code == 200:
                response_data = response.json()
                assistant_message = response_data["choices"][0]["message"]["content"]
                
                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                return assistant_message
            else:
                print(f"Error: API returned status code {response.status_code}")
                print(response.text)
                return "I'm sorry, I'm having trouble connecting to my language model right now."
        
        except Exception as e:
            print(f"Error in processing text with LLM: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def retrieve_relevant_information(self, query):
        """Retrieve relevant information from knowledge base for RAG."""
        # Placeholder for RAG retrieval logic
        print("Retrieving relevant information for RAG...")
        time.sleep(0.5)  # Simulate search time
        return ""
    
    def speak_response(self, text):
        """Convert text to speech and speak it out loud with interrupt handling."""
        if not text:
            return
        
        self.speaking = True
        print(f"Bot: {text}")
        
        # Split text into sentences for better interrupt handling
        # More sophisticated sentence splitting
        sentences = []
        for s in text.split('.'):
            s = s.strip()
            if s:  # Only add non-empty strings
                if len(s) > 80:  # Split very long sentences at commas for better interrupt points
                    comma_splits = s.split(', ')
                    for i, split in enumerate(comma_splits):
                        if i < len(comma_splits) - 1:
                            sentences.append(split + ',')
                        else:
                            sentences.append(split)
                else:
                    sentences.append(s)
        
        for sentence in sentences:
            # Check for interruptions before speaking each sentence
            if not self.interrupt_queue.empty():
                print("Speech interrupted")
                self.interrupt_queue.get()
                self.speaking = False
                break
            
            # Speak the sentence
            self.engine.say(sentence)
            self.engine.runAndWait()
            
            # Brief pause between sentences for more natural speech and better interrupt opportunities
            time.sleep(0.2)
        
        self.speaking = False
    
    def get_audio_energy(self, audio):
        """Calculate energy level from audio data."""
        try:
            # Try to use raw audio data if available
            if hasattr(audio, 'get_raw_data'):
                raw_data = audio.get_raw_data()
                # Convert bytes to numpy array for energy calculation
                data = np.frombuffer(raw_data, dtype=np.int16)
                return np.sqrt(np.mean(np.square(data)))
            else:
                # Fallback to recognizer's energy calculation
                return self.recognizer.energy_threshold * 1.2
        except Exception as e:
            print(f"Energy calculation error: {e}")
            return self.recognizer.energy_threshold * 1.2
    
    def dynamic_interrupt_threshold(self):
        """Dynamically adjust interrupt threshold based on recent energy history."""
        if len(self.interrupt_energy_history) > 0:
            # Average of recent energy readings plus a buffer
            avg_energy = np.mean(self.interrupt_energy_history)
            return avg_energy * self.interrupt_sensitivity
        else:
            # Default fallback based on speech recognizer threshold
            return self.recognizer.energy_threshold * self.interrupt_sensitivity
    
    def interrupt_handler(self):
        """Enhanced handler for user interruptions during speech."""
        while not self.should_stop:
            if self.speaking:
                # Check for voice activation during speaking
                with sr.Microphone() as source:
                    try:
                        # Quick ambient noise adjustment
                        self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                        
                        # Non-blocking listen with short timeout
                        audio = self.recognizer.listen(source, timeout=0.8, phrase_time_limit=1.0)
                        
                        # Calculate audio energy
                        energy = self.get_audio_energy(audio)
                        
                        # Store energy in history (keep last 5 readings)
                        self.interrupt_energy_history.append(energy)
                        if len(self.interrupt_energy_history) > 5:
                            self.interrupt_energy_history.pop(0)
                        
                        # Get dynamic threshold
                        threshold = self.dynamic_interrupt_threshold()
                        
                        # Check if energy exceeds threshold and enough time has passed since last interrupt
                        current_time = time.time()
                        if (energy > threshold and 
                            current_time - self.last_interrupt_time > self.interrupt_cooldown):
                            
                            print(f"Interrupt detected (Energy: {energy:.2f}, Threshold: {threshold:.2f})")
                            self.interrupt_queue.put(True)
                            self.last_interrupt_time = current_time
                            
                            # Attempt to capture what was said during interruption
                            try:
                                interrupted_text = self.recognizer.recognize_google(audio)
                                if interrupted_text:
                                    print(f"Interrupted with: {interrupted_text}")
                                    # Store for potential processing
                                    self.conversation_history.append({"role": "user", "content": interrupted_text})
                            except:
                                pass
                    
                    except (sr.WaitTimeoutError, sr.UnknownValueError):
                        # Expected exceptions during short listening, just continue
                        pass
                    except Exception as e:
                        # Log other exceptions but continue running
                        print(f"Interrupt detection error: {e}")
            
            # Short sleep to reduce CPU usage
            time.sleep(0.1)
    
    def run(self):
        """Run the main conversation loop."""
        # Start the interrupt handler in a separate thread
        interrupt_thread = threading.Thread(target=self.interrupt_handler)
        interrupt_thread.daemon = True
        interrupt_thread.start()
        
        # Initial greeting
        welcome_message = "Hello! I'm Alish's voice assistant. How can I help you today?"
        self.speak_response(welcome_message)
        
        try:
            while not self.should_stop:
                # Get user input through speech
                user_input = self.listen()
                
                if user_input:
                    # Process input and generate response
                    start_time = time.time()
                    response = self.process_with_llm(user_input)
                    processing_time = time.time() - start_time
                    print(f"Response generated in {processing_time:.2f} seconds")
                    
                    # Speak the response
                    self.speak_response(response)
                
                # Small pause between conversation turns
                time.sleep(0.3)
        
        except KeyboardInterrupt:
            print("Stopping the bot...")
        
        finally:
            self.should_stop = True
            print("Bot stopped. Goodbye!")

# UI implementation with enhanced interrupt controls
def create_ui():
    import tkinter as tk
    from tkinter import ttk, Scale
    
    root = tk.Tk()
    root.title("Alish's voice assistant")
    root.geometry("600x700")
    
    # Create and configure frames
    header_frame = ttk.Frame(root, padding=10)
    header_frame.pack(fill=tk.X)
    
    content_frame = ttk.Frame(root, padding=10)
    content_frame.pack(fill=tk.BOTH, expand=True)
    
    control_frame = ttk.Frame(root, padding=10)
    control_frame.pack(fill=tk.X)
    
    # Add title
    title_label = ttk.Label(header_frame, text="Alish's Voice Assistant", font=("Arial", 16))
    title_label.pack()
    
    # Add model selection
    model_frame = ttk.Frame(header_frame)
    model_frame.pack(fill=tk.X, pady=5)
    
    model_label = ttk.Label(model_frame, text="Groq Model:")
    model_label.pack(side=tk.LEFT, padx=5)
    
    model_var = tk.StringVar(value="llama3-70b-8192")
    model_combo = ttk.Combobox(model_frame, textvariable=model_var)
    model_combo['values'] = ('llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it')
    model_combo.pack(side=tk.LEFT, padx=5)
    
    # Add conversation display
    conversation_text = tk.Text(content_frame, height=20, width=50, wrap=tk.WORD)
    conversation_text.pack(fill=tk.BOTH, expand=True)
    conversation_text.config(state=tk.DISABLED)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(content_frame, command=conversation_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    conversation_text.config(yscrollcommand=scrollbar.set)
    
    # Bot instance
    bot = SpeechToSpeechBot(use_rag=False)
    
    # Status indicators
    status_frame = ttk.Frame(root)
    status_frame.pack(fill=tk.X, padx=10, pady=5)
    
    status_var = tk.StringVar(value="Ready")
    status_label = ttk.Label(status_frame, textvariable=status_var)
    status_label.pack(side=tk.LEFT, padx=5)
    
    time_var = tk.StringVar(value="Response time: 0.00s")
    time_label = ttk.Label(status_frame, textvariable=time_var)
    time_label.pack(side=tk.LEFT, padx=5)
    
    # Add advanced interrupt controls
    interrupt_frame = ttk.LabelFrame(root, text="Interrupt Settings", padding=10)
    interrupt_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Sensitivity slider
    sensitivity_label = ttk.Label(interrupt_frame, text="Interrupt Sensitivity:")
    sensitivity_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    
    sensitivity_var = tk.DoubleVar(value=bot.interrupt_sensitivity)
    sensitivity_scale = Scale(interrupt_frame, from_=0.5, to=3.0, resolution=0.1, 
                            orient=tk.HORIZONTAL, variable=sensitivity_var, 
                            length=200, showvalue=True)
    sensitivity_scale.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
    
    def update_sensitivity(event=None):
        bot.interrupt_sensitivity = sensitivity_var.get()
        print(f"Interrupt sensitivity set to: {bot.interrupt_sensitivity}")
    
    sensitivity_scale.bind("<ButtonRelease-1>", update_sensitivity)
    
    # Cooldown slider
    cooldown_label = ttk.Label(interrupt_frame, text="Cooldown (seconds):")
    cooldown_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    
    cooldown_var = tk.DoubleVar(value=bot.interrupt_cooldown)
    cooldown_scale = Scale(interrupt_frame, from_=0.0, to=1.0, resolution=0.1, 
                          orient=tk.HORIZONTAL, variable=cooldown_var, 
                          length=200, showvalue=True)
    cooldown_scale.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W+tk.E)
    
    def update_cooldown(event=None):
        bot.interrupt_cooldown = cooldown_var.get()
        print(f"Interrupt cooldown set to: {bot.interrupt_cooldown} seconds")
    
    cooldown_scale.bind("<ButtonRelease-1>", update_cooldown)
    
    # Manual interrupt button
    def force_interrupt():
        if bot.speaking:
            print("Manual interrupt triggered")
            bot.interrupt_queue.put(True)
    
    interrupt_button = ttk.Button(interrupt_frame, text="Interrupt Now", command=force_interrupt)
    interrupt_button.grid(row=2, column=0, columnspan=2, padx=5, pady=10)
    
    # Energy level indicator
    energy_frame = ttk.Frame(interrupt_frame)
    energy_frame.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
    
    energy_label = ttk.Label(energy_frame, text="Current Energy Level:")
    energy_label.pack(side=tk.LEFT, padx=5)
    
    energy_var = tk.StringVar(value="0")
    energy_value = ttk.Label(energy_frame, textvariable=energy_var)
    energy_value.pack(side=tk.LEFT, padx=5)
    
    threshold_label = ttk.Label(energy_frame, text="Threshold:")
    threshold_label.pack(side=tk.LEFT, padx=5)
    
    threshold_var = tk.StringVar(value="0")
    threshold_value = ttk.Label(energy_frame, textvariable=threshold_var)
    threshold_value.pack(side=tk.LEFT, padx=5)
    
    # Function to update conversation display
    def update_conversation(speaker, text):
        conversation_text.config(state=tk.NORMAL)
        conversation_text.insert(tk.END, f"{speaker}: {text}\n\n")
        conversation_text.see(tk.END)
        conversation_text.config(state=tk.DISABLED)
    
    # Modified bot methods to update UI
    original_speak = bot.speak_response
    original_listen = bot.listen
    original_process = bot.process_with_llm
    
    def ui_speak_response(text):
        status_var.set("Speaking...")
        root.update()
        update_conversation("Bot", text)
        original_speak(text)
        status_var.set("Ready")
        root.update()
    
    def ui_listen():
        status_var.set("Listening...")
        root.update()
        text = original_listen()
        if text:
            update_conversation("You", text)
        status_var.set("Processing...")
        root.update()
        return text
    
    def ui_process_with_llm(text):
        # Update model based on selection
        bot.model = model_var.get()
        
        start_time = time.time()
        response = original_process(text)
        processing_time = time.time() - start_time
        
        time_var.set(f"Response time: {processing_time:.2f}s")
        root.update()
        
        return response
    
    # Replace bot methods
    bot.speak_response = ui_speak_response
    bot.listen = ui_listen
    bot.process_with_llm = ui_process_with_llm
    
    # Function to update energy display
    def update_energy_display():
        if hasattr(bot, 'interrupt_energy_history') and bot.interrupt_energy_history:
            current_energy = bot.interrupt_energy_history[-1] if bot.interrupt_energy_history else 0
            energy_var.set(f"{current_energy:.2f}")
            
            threshold = bot.dynamic_interrupt_threshold() if hasattr(bot, 'dynamic_interrupt_threshold') else 0
            threshold_var.set(f"{threshold:.2f}")
        
        root.after(200, update_energy_display)
    
    # Start energy display updates
    update_energy_display()
    
    # Control buttons
    def start_conversation():
        threading.Thread(target=conversation_loop, daemon=True).start()
    
    def conversation_loop():
        try:
            # Initial greeting
            welcome_message = "Hello! I'm Alish's voice assistant powered by Groq. You can interrupt me anytime."
            bot.speak_response(welcome_message)
            
            while not bot.should_stop:
                user_input = bot.listen()
                
                if user_input:
                    # Process input and generate response
                    response = bot.process_with_llm(user_input)
                    
                    # Speak the response
                    bot.speak_response(response)
                
                time.sleep(0.3)
        except Exception as e:
            print(f"Error in conversation loop: {e}")
    
    def stop_bot():
        bot.should_stop = True
        status_var.set("Stopped")
    
    def toggle_rag():
        bot.use_rag = not bot.use_rag
        if bot.use_rag:
            bot.initialize_rag()
            rag_button.config(text="RAG: ON")
        else:
            rag_button.config(text="RAG: OFF")
    
    # Control buttons frame
    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, padx=10, pady=5)
    
    start_button = ttk.Button(button_frame, text="Start", command=start_conversation)
    start_button.pack(side=tk.RIGHT, padx=5)
    
    stop_button = ttk.Button(button_frame, text="Stop", command=stop_bot)
    stop_button.pack(side=tk.RIGHT, padx=5)
    
    rag_button = ttk.Button(button_frame, text="RAG: OFF", command=toggle_rag)
    rag_button.pack(side=tk.RIGHT, padx=5)
    
    # API key configuration
    settings_frame = ttk.LabelFrame(root, text="API Settings", padding=10)
    settings_frame.pack(fill=tk.X, padx=10, pady=5)
    
    api_label = ttk.Label(settings_frame, text="Groq API Key:")
    api_label.pack(side=tk.LEFT, padx=5)
    
    api_var = tk.StringVar(value=os.getenv("GROQ_API_KEY", ""))
    api_entry = ttk.Entry(settings_frame, textvariable=api_var, width=40, show="*")
    api_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def save_api_key():
        bot.api_key = api_var.get()
        os.environ["GROQ_API_KEY"] = api_var.get()
        print("API key updated")
    
    save_button = ttk.Button(settings_frame, text="Save", command=save_api_key)
    save_button.pack(side=tk.RIGHT, padx=5)
    
    # Start UI
    root.mainloop()

if __name__ == "__main__":
    # Run with UI
    create_ui()
    
   