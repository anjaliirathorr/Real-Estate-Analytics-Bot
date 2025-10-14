// // Voice recognition functionality
// document.getElementById("mic-btn").addEventListener("click", () => {
//     const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
//     recognition.lang = "en-US";
//     recognition.start();
//     recognition.onresult = (event) => {
//         const transcript = event.results[0][0].transcript;
//         addUserMessage(transcript);
//     };
//     recognition.onerror = (event) => {
//         alert(`Error: ${event.error}`);
//     };
// });

// // Text input functionality
// document.getElementById("send-btn").addEventListener("click", sendMessage);
// document.getElementById("user-input").addEventListener("keypress", (e) => {
//     if (e.key === "Enter") {
//         sendMessage();
//     }
// });

// function sendMessage() {
//     const userInput = document.getElementById("user-input");
//     const message = userInput.value.trim();
    
//     if (message) {
//         addUserMessage(message);
//         userInput.value = "";
//     }
// }

// function addUserMessage(text) {
//     const chatContainer = document.getElementById("chat-container");
//     const userMessage = document.createElement("div");
//     userMessage.classList.add("message", "user");
//     userMessage.textContent = text;
//     chatContainer.appendChild(userMessage);
//     chatContainer.scrollTop = chatContainer.scrollHeight;
    
//     // Add a loading message
//     const loadingMessage = document.createElement("div");
//     loadingMessage.classList.add("message", "agent");
//     loadingMessage.textContent = "Processing your request...";
//     loadingMessage.id = "loading-message";
//     chatContainer.appendChild(loadingMessage);
//     chatContainer.scrollTop = chatContainer.scrollHeight;
    
//     // Make API call to the backend
//     fetch('/api/query', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ query: text })
//     })
//     .then(response => {
//         if (!response.ok) {
//             throw new Error('Network response was not ok');
//         }
//         return response.json();
//     })
//     .then(data => {
//         // Remove the loading message
//         document.getElementById("loading-message").remove();
        
//         // Add the actual response
//         const agentMessage = document.createElement("div");
//         agentMessage.classList.add("message", "agent");
        
//         // Format the response for better readability
//         const formattedResponse = formatResponseText(data.response);
//         agentMessage.innerHTML = formattedResponse;
        
//         chatContainer.appendChild(agentMessage);
//         chatContainer.scrollTop = chatContainer.scrollHeight;
        
//         // Speak the response if audio is enabled
//         speakResponse(data.response);
//     })
//     .catch(error => {
//         // Remove the loading message
//         if (document.getElementById("loading-message")) {
//             document.getElementById("loading-message").remove();
//         }
        
//         // Show error message
//         const errorMessage = document.createElement("div");
//         errorMessage.classList.add("message", "agent");
//         errorMessage.textContent = "Sorry, there was an error processing your request. Please try again.";
//         chatContainer.appendChild(errorMessage);
//         chatContainer.scrollTop = chatContainer.scrollHeight;
//         console.error('Error:', error);
//     });
// }

// // Format the response text with proper line breaks and highlights
// function formatResponseText(text) {
//     // Replace property titles with highlighted versions
//     let formattedText = text.replace(/Property \d+:/g, match => `<strong>${match}</strong>`);
    
//     // Replace line breaks with HTML breaks
//     formattedText = formattedText.replace(/\n/g, '<br>');
    
//     // Highlight important information
//     formattedText = formattedText.replace(/- ([^:]+):/g, '- <strong>$1</strong>:');
    
//     return formattedText;
// }

// // Audio output functionality
// let isSpeaking = false;
// const synth = window.speechSynthesis;

// document.getElementById("speaker-btn").addEventListener("click", () => {
//     isSpeaking = !isSpeaking;
//     const speakerBtn = document.getElementById("speaker-btn");
//     speakerBtn.textContent = isSpeaking ? "🔇" : "🔊";
    
//     if (!isSpeaking) {
//         synth.cancel();
//     }
// });

// function speakResponse(text) {
//     if (!isSpeaking) return;
    
//     // Clean up the text for speaking (remove property details)
//     const cleanText = text.split("Property 1:")[0];
    
//     const utterance = new SpeechSynthesisUtterance(cleanText);
//     utterance.rate = 1.0;
//     utterance.pitch = 1.0;
//     synth.speak(utterance);
// }


















document.addEventListener('DOMContentLoaded', () => {
    // Check if speech recognition is available
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
        document.getElementById("mic-btn").style.display = "none";
        console.warn("Speech recognition not supported in this browser");
    }
    
    // Initialize with a greeting
    addAgentMessage("Hello! 👋 I'm your personal real estate assistant. How can I help you find your perfect property today?");
    
    // Initialize icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
    
    // Show chat widget initially (for demo, you may want to change this to 'none' for production)
    document.getElementById("chat-widget-dialog").style.display = 'none';
    document.getElementById("chatbot-toggle").style.display = 'block';

    // Chat toggle functionality
    const toggleBtn = document.getElementById('chatbot-toggle');
    const chatWidget = document.getElementById('chat-widget-dialog');
    const closeBtn = document.getElementById('close-chat');
          
    toggleBtn.addEventListener('click', () => {
        chatWidget.style.display = 'flex';
        toggleBtn.style.display = 'none';
    });
          
    closeBtn.addEventListener('click', () => {
        chatWidget.style.display = 'none';
        toggleBtn.style.display = 'flex';
    });

    // Add this block for query recommendations
    const queryButtons = document.querySelectorAll(".query-btn");
    const userInput = document.getElementById("user-input");

    queryButtons.forEach((button) => {
        button.addEventListener("click", () => {
            userInput.value = button.textContent; // Set the input box value to the query text
            userInput.focus(); // Focus on the input box
        });
    });

    // Load voices when the page loads
    ensureVoicesLoaded(voices => {
        console.log("Available voices:", voices.map(v => v.name));
    });
});

// Global variables for speech functionality
let isListening = false;
let isSpeaking = false;
let recognition = null;
const synth = window.speechSynthesis;
let femaleVoice = null;

// Voice recognition functionality
document.getElementById("mic-btn").addEventListener("click", toggleSpeechRecognition);

function toggleSpeechRecognition() {
    if (isListening) {
        stopListening();
    } else {
        startListening();
    }
}

function startListening() {
    // Create recognition object if it doesn't exist
    if (!recognition) {
        recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        recognition.lang = "en-US";
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            document.getElementById("user-input").value = transcript;
            // Small delay to show the transcribed text before sending
            setTimeout(() => {
                sendMessage();
            }, 500);
        };
        
        recognition.onerror = (event) => {
            console.error(`Speech recognition error: ${event.error}`);
            stopListening();
        };
        
        recognition.onend = () => {
            if (isListening) {
                stopListening();
            }
        };
    }
    
    isListening = true;
    recognition.start();
    
    // Update mic button appearance
    const micBtn = document.getElementById("mic-btn");
    micBtn.innerHTML = '<i data-lucide="mic-off"></i>';
    micBtn.classList.add("active");
    updateIcons();
}

function stopListening() {
    if (recognition) {
        recognition.stop();
    }
    isListening = false;
    
    // Update mic button appearance
    const micBtn = document.getElementById("mic-btn");
    micBtn.innerHTML = '<i data-lucide="mic"></i>';
    micBtn.classList.remove("active");
    updateIcons();
}

// Text-to-speech functionality
document.getElementById("speaker-btn").addEventListener("click", toggleSpeaking);

function toggleSpeaking() {
    isSpeaking = !isSpeaking;
    
    // Update speaker button appearance
    const speakerBtn = document.getElementById("speaker-btn");
    
    if (isSpeaking) {
        speakerBtn.innerHTML = '<i data-lucide="volume-2"></i>';
        speakerBtn.classList.add("active");
        
        // Speak the last agent message if available
        const messages = document.querySelectorAll('.message.agent');
        if (messages.length > 0) {
            const lastMessage = messages[messages.length - 1];
            speakResponse(lastMessage.innerText);
        }
    } else {
        speakerBtn.innerHTML = '<i data-lucide="volume-x"></i>';
        speakerBtn.classList.remove("active");
        synth.cancel(); // Stop any ongoing speech
    }
    
    updateIcons();
}

function speakResponse(text) {
    if (!isSpeaking) return;
    
    // Cancel any ongoing speech
    synth.cancel();
    
    // Clean up the text for speaking
    let cleanText = text.replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold markers
                       .replace(/\*(.*?)\*/g, '$1')    // Remove italic markers
                       .replace(/<br>/g, ' ')          // Replace HTML breaks with spaces
                       .replace(/🏠/g, 'Property');    // Replace property emoji
    
    // Get only the first part of the message before detailed listings
    if (cleanText.includes("Property 1:")) {
        cleanText = cleanText.split("Property 1:")[0] + "I found some properties that match your criteria.";
    }
    
    // Create utterance
    const utterance = new SpeechSynthesisUtterance(cleanText);
    
    // Set voice properties
    utterance.rate = 1.0;
    utterance.pitch = 1.1; // Slightly higher pitch for female voice
    utterance.volume = 1.0;
    
    // Add event listeners for speech feedback
    utterance.onstart = () => {
        const speakerBtn = document.getElementById("speaker-btn");
        speakerBtn.classList.add("speaking");
    };
    
    utterance.onend = () => {
        const speakerBtn = document.getElementById("speaker-btn");
        speakerBtn.classList.remove("speaking");
    };
    
    // Speak the text
    synth.speak(utterance);
}

// Text input functionality
document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        sendMessage();
    }
});

function sendMessage() {
    const userInput = document.getElementById("user-input");
    const message = userInput.value.trim();
    
    if (message) {
        addUserMessage(message);
        userInput.value = "";
        userInput.focus();
        
        // Stop listening if it's active
        if (isListening) {
            recognition.stop();
        }
        
        // Call the backend API instead of using hardcoded responses
        callChatApi(message);
    }
}

// Function to call the actual backend API
async function callChatApi(query) {
    // Add a loading message
    const chatContainer = document.getElementById("chat-container");
    const loadingMessage = document.createElement("div");
    loadingMessage.classList.add("message", "agent", "loading");
    loadingMessage.textContent = "Processing your request";
    loadingMessage.id = "loading-message";
    chatContainer.appendChild(loadingMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    try {
        // Call the Flask backend
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: query })
        });
        
        // Remove the loading message
        const loadingEl = document.getElementById("loading-message");
        if (loadingEl) loadingEl.remove();
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Show acknowledgment if provided
        if (data.acknowledgment) {
            const ackMessage = document.createElement("div");
            ackMessage.classList.add("message", "agent", "acknowledgment");
            ackMessage.textContent = data.acknowledgment;
            chatContainer.appendChild(ackMessage);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            // Remove acknowledgment after the full response arrives
            setTimeout(() => {
                ackMessage.remove();
            }, 1500);
        }
        
        // Add the full response to the chat
        addAgentMessage(data.message);
        
        // Speak the response if audio is enabled
        if (isSpeaking) {
            speakResponse(data.message);
        }
    } catch (error) {
        console.error('Error:', error);
        
        // Remove the loading message
        const loadingEl = document.getElementById("loading-message");
        if (loadingEl) loadingEl.remove();
        
        // Add error message
        addAgentMessage("Sorry, I encountered an error processing your request. Please try again.");
    }
}

// Fallback function if the API fails
function generateFallbackResponse(query) {
    query = query.toLowerCase();
    
    if (query.includes('hello') || query.includes('hi') || query.includes('hey')) {
        return "Hi there! 👋 How can I help you with your property search today?";
    }
    else if (query.includes('buy') || query.includes('looking for') || query.includes('search')) {
        return "I'd be happy to help you find a property! Could you tell me more about what you're looking for?";
    }
    else if (query.includes('sell')) {
        return "Interested in selling your property? Great! I can help you with that.";
    }
    else if (query.includes('price') || query.includes('cost') || query.includes('budget')) {
        return "Property prices vary depending on location, size, and amenities. I'll search for properties in your budget.";
    }
    else {
        return "I'll help you find the perfect property. What specific features are you looking for?";
    }
}

function addUserMessage(text) {
    const chatContainer = document.getElementById("chat-container");
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "user");
    userMessage.textContent = text;
    chatContainer.appendChild(userMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addAgentMessage(text) {
    const chatContainer = document.getElementById("chat-container");
    const agentMessage = document.createElement("div");
    agentMessage.classList.add("message", "agent");
    
    // Format the response for better readability
    const formattedResponse = formatResponseText(text);
    agentMessage.innerHTML = formattedResponse;
    
    chatContainer.appendChild(agentMessage);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format the response text with proper line breaks and highlights
function formatResponseText(text) {
    // Convert markdown style formatting
    let formattedText = text;
    
    // Replace markdown bold with HTML strong
    formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Replace markdown italic with HTML em
    formattedText = formattedText.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Replace line breaks with HTML breaks
    formattedText = formattedText.replace(/\n/g, '<br>');
    
    // Highlight emojis with a slight emphasis
    const emojiRegex = /(?:[\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g;
    formattedText = formattedText.replace(emojiRegex, '<span class="emoji">$&</span>');
    
    return formattedText;
}

// Ensure Lucide icons are updated
function updateIcons() {
    if (typeof lucide !== 'undefined' && lucide.createIcons) {
        lucide.createIcons();
    }
}