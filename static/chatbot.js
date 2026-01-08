// Chatbot functionality
class DiabetesChatbot {
    constructor() {
        this.isOpen = false;
        this.messages = [];
        this.init();
    }

    init() {
        this.createChatbotHTML();
        this.attachEventListeners();
        this.addWelcomeMessage();
    }

    createChatbotHTML() {
        const chatbotHTML = `
            <div class="chatbot-container" id="chatbot">
                <div class="chatbot-header">
                    <div class="chatbot-header-content">
                        <span class="chatbot-icon">🤖</span>
                        <span class="chatbot-title">Diabetes Assistant</span>
                        <span class="chatbot-status">Online</span>
                    </div>
                    <button class="chatbot-close" id="closeChatbot">✕</button>
                </div>
                <div class="chatbot-messages" id="chatbotMessages"></div>
                <div class="chatbot-input-container">
                    <input 
                        type="text" 
                        class="chatbot-input" 
                        id="chatbotInput" 
                        placeholder="Type your question..."
                        autocomplete="off"
                    />
                    <button class="chatbot-send" id="sendMessage">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
                <div class="chatbot-quick-questions">
                    <button class="quick-question" data-question="What is diabetes?">What is diabetes?</button>
                    <button class="quick-question" data-question="How does this prediction work?">How does it work?</button>
                    <button class="quick-question" data-question="What are normal glucose levels?">Normal levels?</button>
                </div>
            </div>
            <button class="chatbot-toggle" id="chatbotToggle">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span class="chatbot-badge" id="chatbotBadge">1</span>
            </button>
        `;
        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
    }

    attachEventListeners() {
        const toggle = document.getElementById('chatbotToggle');
        const close = document.getElementById('closeChatbot');
        const send = document.getElementById('sendMessage');
        const input = document.getElementById('chatbotInput');
        const quickQuestions = document.querySelectorAll('.quick-question');

        toggle.addEventListener('click', () => this.toggleChat());
        close.addEventListener('click', () => this.toggleChat());
        send.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });

        quickQuestions.forEach(btn => {
            btn.addEventListener('click', () => {
                const question = btn.getAttribute('data-question');
                this.sendMessage(question);
            });
        });
    }

    toggleChat() {
        this.isOpen = !this.isOpen;
        const chatbot = document.getElementById('chatbot');
        const toggle = document.getElementById('chatbotToggle');
        const badge = document.getElementById('chatbotBadge');

        if (this.isOpen) {
            chatbot.classList.add('open');
            toggle.classList.add('active');
            badge.style.display = 'none';
        } else {
            chatbot.classList.remove('open');
            toggle.classList.remove('active');
        }
    }

    addWelcomeMessage() {
        const welcomeMsg = "Hi! 👋 I'm your Diabetes Assistant. I can help you understand diabetes, guide you through predictions, and answer your health questions. How can I help you today?";
        this.addMessage(welcomeMsg, 'bot');
    }

    addMessage(text, sender = 'user') {
        const messagesContainer = document.getElementById('chatbotMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chatbot-message ${sender}-message`;
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            <div class="message-content">${text}</div>
            <div class="message-time">${time}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        this.messages.push({ text, sender, time });
    }

    async sendMessage(customMessage = null) {
        const input = document.getElementById('chatbotInput');
        const message = customMessage || input.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage(message, 'user');
        if (!customMessage) input.value = '';

        // Show typing indicator
        this.showTyping();

        try {
            const response = await fetch('/chatbot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();
            
            // Remove typing indicator
            this.removeTyping();
            
            if (data.response) {
                this.addMessage(data.response, 'bot');
            } else {
                this.addMessage("I'm sorry, I couldn't process that. Please try again.", 'bot');
            }
        } catch (error) {
            this.removeTyping();
            this.addMessage("Sorry, I'm having trouble connecting. Please try again later.", 'bot');
        }
    }

    showTyping() {
        const messagesContainer = document.getElementById('chatbotMessages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'chatbot-message bot-message typing-indicator';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="message-content">
                <span></span><span></span><span></span>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    removeTyping() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new DiabetesChatbot();
});
