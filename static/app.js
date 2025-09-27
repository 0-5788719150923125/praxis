// Praxis Chat - Main Application JavaScript

// ==================== Global Variables ====================
let currentTheme = 'light';
let conversationHistory = [];
let generationParams = {
    max_new_tokens: 256,
    temperature: 0.5,
    repetition_penalty: 1.2,
    do_sample: true,
    use_cache: false
};
let debugLogging = false;
let terminalSocket = null;
let terminalConnected = false;
let dashboardScale = null;
let currentTab = 'chat';
let currentFrameContainer = null;
let currentWrapperDiv = null;
let terminalReconnectTimeout = null;
let terminalReconnectAttempts = 0;
let lastFrameHash = null;
let frameStuckCounter = 0;
let lastFrameUpdateTime = Date.now();
let frameValidationInterval = null;
let specLoaded = false;
let agentsLoaded = false;
let agentsRefreshInterval = null;
let isShowingPlaceholder = true;

// ==================== Constants ====================
const MAX_HISTORY_LENGTH = 21;
const PREFIX = "> ";
const PLACEHOLDER_TEXT = "Shoot";
const SYSTEM_PROMPT = "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation.";

// Theme icons
const sunIcon = `<path d="M8 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6zm0 1a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM8 0a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 0zm0 13a.5.5 0 0 1 .5.5v2a.5.5 0 0 1-1 0v-2A.5.5 0 0 1 8 13zm8-5a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2a.5.5 0 0 1 .5.5zM3 8a.5.5 0 0 1-.5.5h-2a.5.5 0 0 1 0-1h2A.5.5 0 0 1 3 8zm10.657-5.657a.5.5 0 0 1 0 .707l-1.414 1.415a.5.5 0 1 1-.707-.708l1.414-1.414a.5.5 0 0 1 .707 0zm-9.193 9.193a.5.5 0 0 1 0 .707L3.05 13.657a.5.5 0 0 1-.707-.707l1.414-1.414a.5.5 0 0 1 .707 0zm9.193 2.121a.5.5 0 0 1-.707 0l-1.414-1.414a.5.5 0 0 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .707zM4.464 4.465a.5.5 0 0 1-.707 0L2.343 3.05a.5.5 0 1 1 .707-.707l1.414 1.414a.5.5 0 0 1 0 .708z"/>`;

const moonIcon = `<path d="M6 .278a.768.768 0 0 1 .08.858 7.208 7.208 0 0 0-.878 3.46c0 4.021 3.278 7.277 7.318 7.277.527 0 1.04-.055 1.533-.16a.787.787 0 0 1 .81.316.733.733 0 0 1-.031.893A8.349 8.349 0 0 1 8.344 16C3.734 16 0 12.286 0 7.71 0 4.266 2.114 1.312 5.124.06A.752.752 0 0 1 6 .278z"/>`;

// API configuration
let pathPrefix = window.location.pathname;
if (pathPrefix.endsWith('/') && pathPrefix !== '/') {
    pathPrefix = pathPrefix.slice(0, -1);
}
if (pathPrefix === '/') {
    pathPrefix = '';
}
const API_BASE_URL = window.location.origin + pathPrefix;
let apiUrl = API_BASE_URL + '/messages/';

// ==================== Live Reload Setup ====================
function setupLiveReload() {
    console.log('Connecting live-reload WebSocket');
    
    const pathname = window.location.pathname;
    let socketPath = '/socket.io';
    
    if (pathname && pathname !== '/') {
        const cleanPath = pathname.replace(/\/$/, '');
        socketPath = cleanPath + '/socket.io';
    }
    
    console.log('Live-reload socket path:', socketPath);
    
    const socket = io.connect('/live-reload', {
        path: socketPath
    });
    
    socket.on('connect', () => {
        console.log('Live reload connected');
    });
    
    socket.on('reload', () => {
        console.log('Template change detected, reloading...');
        window.location.reload();
    });
    
    socket.on('disconnect', () => {
        console.log('Live reload disconnected');
    });
}

// ==================== Theme Management ====================
function setTheme(theme) {
    currentTheme = theme;
    const themeIcon = document.getElementById('theme-icon');
    
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeIcon.innerHTML = sunIcon;
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeIcon.innerHTML = moonIcon;
    }
    localStorage.setItem('praxis_theme', theme);
}

function toggleTheme() {
    setTheme(currentTheme === 'light' ? 'dark' : 'light');
    
    // Update all message headers when theme changes
    const chatContainer = document.getElementById('chat-container');
    const messages = chatContainer.querySelectorAll('.message');
    messages.forEach(message => {
        const header = message.querySelector('.message-header');
        if (header) {
            const isUser = message.classList.contains('user');
            const isDarkMode = currentTheme === 'dark';
            if (isDarkMode) {
                header.textContent = isUser ? 'Me' : 'You';
            } else {
                header.textContent = isUser ? 'You' : 'Me';
            }
        }
    });
}

function loadTheme() {
    const savedTheme = localStorage.getItem('praxis_theme') || 'light';
    setTheme(savedTheme);
    
    // Update existing message headers based on loaded theme
    const chatContainer = document.getElementById('chat-container');
    const messages = chatContainer.querySelectorAll('.message');
    messages.forEach(message => {
        const header = message.querySelector('.message-header');
        if (header) {
            const isUser = message.classList.contains('user');
            const isDarkMode = savedTheme === 'dark';
            if (isDarkMode) {
                header.textContent = isUser ? 'Me' : 'You';
            } else {
                header.textContent = isUser ? 'You' : 'Me';
            }
        }
    });
}

// ==================== Settings Management ====================
function loadSettings() {
    loadTheme();
    
    // Load developer prompt
    const developerPromptElement = document.getElementById('developer-prompt');
    const savedDeveloperPrompt = localStorage.getItem('praxis_developer_prompt');
    if (savedDeveloperPrompt) {
        developerPromptElement.innerHTML = savedDeveloperPrompt;
    }
    
    // Load API URL
    const apiUrlInput = document.getElementById('api-url');
    const savedApiUrl = localStorage.getItem('praxis_api_url');
    if (savedApiUrl) {
        if (window.location.hostname.includes('ngrok') && pathPrefix !== '') {
            apiUrlInput.value = apiUrl;
            console.log('Using dynamic ngrok URL instead of saved URL');
        } else {
            apiUrl = savedApiUrl;
            apiUrlInput.value = savedApiUrl;
        }
    } else {
        apiUrlInput.value = apiUrl;
    }
    
    // Load generation parameters
    const savedParams = localStorage.getItem('praxis_gen_params');
    if (savedParams) {
        try {
            const params = JSON.parse(savedParams);
            generationParams = { ...generationParams, ...params };
            
            // Update UI elements
            document.getElementById('max-tokens').value = generationParams.max_new_tokens;
            document.getElementById('temperature').value = generationParams.temperature;
            document.getElementById('temperature-value').textContent = generationParams.temperature;
            document.getElementById('repetition-penalty').value = generationParams.repetition_penalty;
            document.getElementById('repetition-penalty-value').textContent = generationParams.repetition_penalty;
            document.getElementById('do-sample').checked = generationParams.do_sample;
        } catch (error) {
            console.error('Error loading saved generation parameters:', error);
        }
    }
    
    // Load debug logging setting
    const savedDebugLogging = localStorage.getItem('praxis_debug_logging');
    if (savedDebugLogging === 'true') {
        debugLogging = true;
        document.getElementById('debug-logging').checked = true;
    }
}

async function testApiConnection(url) {
    try {
        let baseUrl = url;
        
        if (baseUrl.endsWith('/messages/')) {
            baseUrl = baseUrl.substring(0, baseUrl.length - 10);
        } else if (baseUrl.endsWith('/input/')) {
            // Handle old saved URLs
            baseUrl = baseUrl.substring(0, baseUrl.length - 7);
        }
        
        if (!baseUrl.startsWith('http://') && !baseUrl.startsWith('https://')) {
            baseUrl = 'http://' + baseUrl;
        }
        
        if (baseUrl.endsWith('/')) {
            baseUrl = baseUrl.substring(0, baseUrl.length - 1);
        }
        
        const pingUrl = `${baseUrl}/api/ping`;
        console.log(`Testing API connection to: ${pingUrl}`);
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Connection test timed out')), 15000)
        );
        
        const fetchPromise = fetch(pingUrl, {
            method: 'GET',
            mode: 'cors',
            credentials: 'omit'
        });
        
        const response = await Promise.race([fetchPromise, timeoutPromise]);
        
        if (!response.ok) {
            return { success: false, message: `Server returned ${response.status} ${response.statusText}` };
        }
        
        let result;
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            result = await response.json();
        } else {
            const text = await response.text();
            console.log('Non-JSON response:', text);
            result = { message: 'Connected (non-JSON response)' };
        }
        
        console.log('API connection test result:', result);
        return { success: true, message: result.message || 'Connected to API server' };
    } catch (error) {
        console.error('API connection test failed:', error);
        if (error.message.includes('Failed to fetch')) {
            return { success: false, message: 'Could not reach the server. Check if it\'s running.' };
        }
        return { success: false, message: 'Connection failed: ' + error.message };
    }
}

async function saveSettings() {
    const apiUrlInput = document.getElementById('api-url');
    const newApiUrl = apiUrlInput.value.trim();
    
    if (newApiUrl) {
        const connectionTest = await testApiConnection(newApiUrl);
        
        // Save generation parameters
        generationParams = {
            max_new_tokens: parseInt(document.getElementById('max-tokens').value, 10),
            temperature: parseFloat(document.getElementById('temperature').value),
            repetition_penalty: parseFloat(document.getElementById('repetition-penalty').value),
            do_sample: document.getElementById('do-sample').checked
        };
        
        localStorage.setItem('praxis_gen_params', JSON.stringify(generationParams));
        
        // Save debug logging preference
        debugLogging = document.getElementById('debug-logging').checked;
        localStorage.setItem('praxis_debug_logging', debugLogging.toString());
        
        if (connectionTest.success) {
            apiUrl = newApiUrl;
            localStorage.setItem('praxis_api_url', newApiUrl);
            
            const confirmationDiv = document.getElementById('save-confirmation');
            confirmationDiv.textContent = 'Connection successful! Settings saved. Refreshing...';
            confirmationDiv.classList.add('show');
            
            setTimeout(() => window.location.reload(), 2000);
        } else {
            console.warn('API connection warning:', connectionTest.message);
            apiUrl = newApiUrl;
            localStorage.setItem('praxis_api_url', newApiUrl);
            
            const confirmationDiv = document.getElementById('save-confirmation');
            confirmationDiv.textContent = 'Settings saved (check console for connection details)';
            confirmationDiv.classList.add('show');
            
            setTimeout(() => window.location.reload(), 2000);
        }
    } else {
        // Save only generation parameters
        generationParams = {
            max_new_tokens: parseInt(document.getElementById('max-tokens').value, 10),
            temperature: parseFloat(document.getElementById('temperature').value),
            repetition_penalty: parseFloat(document.getElementById('repetition-penalty').value),
            do_sample: document.getElementById('do-sample').checked
        };
        
        localStorage.setItem('praxis_gen_params', JSON.stringify(generationParams));
        
        debugLogging = document.getElementById('debug-logging').checked;
        localStorage.setItem('praxis_debug_logging', debugLogging.toString());
        
        const confirmationDiv = document.getElementById('save-confirmation');
        confirmationDiv.textContent = 'Generation parameters saved!';
        confirmationDiv.classList.add('show');
        
        setTimeout(() => {
            confirmationDiv.classList.remove('show');
        }, 2000);
    }
}

// ==================== Modal Management ====================
function openModal() {
    document.getElementById('settings-modal').classList.add('open');
}

function closeModal() {
    document.getElementById('settings-modal').classList.remove('open');
}

// ==================== Message Input Management ====================
function setCursorAfterPrefix() {
    const messageInput = document.getElementById('message-input');
    messageInput.setSelectionRange(PREFIX.length, PREFIX.length);
}

function showPlaceholder() {
    const messageInput = document.getElementById('message-input');
    if (messageInput.value === PREFIX) {
        messageInput.value = PREFIX + PLACEHOLDER_TEXT;
        messageInput.style.color = 'var(--light-text)';
        messageInput.style.fontStyle = 'italic';
        isShowingPlaceholder = true;
        setCursorAfterPrefix();
    }
}

function hidePlaceholder() {
    const messageInput = document.getElementById('message-input');
    if (isShowingPlaceholder) {
        messageInput.value = PREFIX;
        messageInput.style.color = '';
        messageInput.style.fontStyle = '';
        isShowingPlaceholder = false;
        setCursorAfterPrefix();
    }
}

function maintainPrefix() {
    const messageInput = document.getElementById('message-input');
    const currentValue = messageInput.value;
    const cursorPos = messageInput.selectionStart;
    
    if (isShowingPlaceholder) {
        const newChars = currentValue.replace(PREFIX + PLACEHOLDER_TEXT, '').replace(PREFIX, '');
        hidePlaceholder();
        if (newChars) {
            messageInput.value = PREFIX + newChars;
            messageInput.setSelectionRange(messageInput.value.length, messageInput.value.length);
        }
        return;
    }
    
    if (!currentValue.startsWith(PREFIX)) {
        const userText = currentValue.replace(/^[>\s]*/, '');
        messageInput.value = PREFIX + userText;
        
        const newCursorPos = Math.max(PREFIX.length, cursorPos + PREFIX.length - (currentValue.length - userText.length));
        messageInput.setSelectionRange(newCursorPos, newCursorPos);
    }
    
    autoResizeTextarea();
}

function autoResizeTextarea() {
    const messageInput = document.getElementById('message-input');
    messageInput.style.height = 'auto';
    messageInput.style.height = messageInput.scrollHeight + 'px';
}

// ==================== Chat Message Management ====================
function addMessage(content, isUser) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
    
    // Add with-messages class to input container
    const inputContainer = document.querySelector('.input-container');
    if (inputContainer && !inputContainer.classList.contains('with-messages')) {
        inputContainer.classList.add('with-messages');
    }
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    // Check if content is empty for assistant messages
    if (!isUser && (!content || content.trim() === '')) {
        headerDiv.innerHTML = '<span class="error-header">[ERR]</span>';
        messageDiv.appendChild(headerDiv);
    } else {
        const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
        if (isDarkMode) {
            headerDiv.textContent = isUser ? 'Me' : 'You';
        } else {
            headerDiv.textContent = isUser ? 'You' : 'Me';
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(headerDiv);
        messageDiv.appendChild(contentDiv);
    }
    
    // Add reroll button for assistant messages
    if (!isUser) {
        const rerollButton = document.createElement('button');
        rerollButton.className = 'reroll-button';
        rerollButton.innerHTML = `
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M1 4v6h6M23 20v-6h-6"/>
                <path d="M20.49 9A9 9 0 1 0 5.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 0 1 3.51 15"/>
            </svg>
            Retry
        `;
        rerollButton.onclick = handleReroll;
        messageDiv.appendChild(rerollButton);
    }
    
    chatContainer.appendChild(messageDiv);
    
    // Smooth scroll to bottom
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
    
    // Add to conversation history
    conversationHistory.push({
        role: isUser ? 'user' : 'assistant',
        content: content
    });
}

function showThinking() {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'thinking-indicator';
    
    const headerDiv = document.createElement('div');
    headerDiv.className = 'message-header';
    
    const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';
    
    let headerContent;
    if (isDarkMode) {
        headerContent = `
            <span class="header-name">You</span>
            <span class="thinking-status"> are thinking</span>
        `;
    } else {
        headerContent = `
            <span class="header-name">Meandering</span>
        `;
    }
    
    headerDiv.innerHTML = headerContent + `
        <div class="dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    
    messageDiv.appendChild(headerDiv);
    
    // Add timer to update thinking indicator
    let thinkingTimeSeconds = 0;
    const thinkingTimer = setInterval(() => {
        thinkingTimeSeconds += 1;
        
        if (thinkingTimeSeconds >= 60) {
            const minutes = Math.floor(thinkingTimeSeconds / 60);
            const seconds = thinkingTimeSeconds % 60;
            const timeDisplay = `${minutes}m ${seconds}s`;
            
            const timerElement = document.createElement('span');
            timerElement.className = 'thinking-timer';
            timerElement.textContent = ` (${timeDisplay})`;
            timerElement.style.fontSize = '12px';
            timerElement.style.color = 'var(--light-text)';
            
            const existingTimer = headerDiv.querySelector('.thinking-timer');
            if (existingTimer) {
                existingTimer.textContent = ` (${timeDisplay})`;
            } else {
                headerDiv.querySelector('.thinking-status').appendChild(timerElement);
            }
        }
    }, 1000);
    
    messageDiv.dataset.timerId = thinkingTimer;
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function hideThinking() {
    const thinkingDiv = document.getElementById('thinking-indicator');
    if (thinkingDiv) {
        if (thinkingDiv.dataset.timerId) {
            clearInterval(parseInt(thinkingDiv.dataset.timerId));
        }
        thinkingDiv.remove();
    }
}

async function handleReroll() {
    let lastUserMessage = null;
    for (let i = conversationHistory.length - 1; i >= 0; i--) {
        if (conversationHistory[i].role === 'user') {
            lastUserMessage = conversationHistory[i].content;
            break;
        }
    }
    
    if (!lastUserMessage) return;
    
    const chatContainer = document.getElementById('chat-container');
    const assistantMessages = chatContainer.querySelectorAll('.message.assistant');
    if (assistantMessages.length > 0) {
        const lastAssistantMessage = assistantMessages[assistantMessages.length - 1];
        lastAssistantMessage.remove();
    }
    
    if (conversationHistory[conversationHistory.length - 1].role === 'assistant') {
        conversationHistory.pop();
    }
    
    await sendMessageToAPI(lastUserMessage);
}

// ==================== API Communication ====================
async function sendMessage() {
    if (isShowingPlaceholder) return;
    
    const messageInput = document.getElementById('message-input');
    const fullValue = messageInput.value;
    
    const message = fullValue.startsWith(PREFIX) 
        ? fullValue.slice(PREFIX.length).trim() 
        : fullValue.trim();
    
    if (!message) return;
    
    addMessage(message, true);
    
    messageInput.value = PREFIX;
    messageInput.style.color = '';
    messageInput.style.fontStyle = '';
    isShowingPlaceholder = false;
    setCursorAfterPrefix();
    
    await sendMessageToAPI(message);
}

async function sendMessageToAPI(message) {
    showThinking();
    
    try {
        const developerPrompt = document.getElementById('developer-prompt').innerText.trim() || "Write thy wrong.";
        const staticPrompts = [
            { role: "system", content: SYSTEM_PROMPT },
            { role: "developer", content: developerPrompt }
        ];
        
        const truncatedHistory = conversationHistory.slice(-MAX_HISTORY_LENGTH);
        const fullMessages = [...staticPrompts, ...truncatedHistory];
        
        const requestBody = {
            messages: fullMessages,
            ...generationParams
        };
        
        if (debugLogging) {
            console.group('🤖 AI Request Debug');
            console.log('Request URL:', apiUrl);
            console.log('Full Request Payload:', JSON.parse(JSON.stringify(requestBody)));
            console.log('Messages Array:');
            requestBody.messages.forEach((msg, index) => {
                console.log(`[${index}] ${msg.role}:`, msg.content);
            });
            console.log('Generation Parameters:', {
                max_new_tokens: requestBody.max_new_tokens,
                temperature: requestBody.temperature,
                repetition_penalty: requestBody.repetition_penalty,
                do_sample: requestBody.do_sample
            });
            console.groupEnd();
        }
        
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Request timed out after 5 minutes')), 5 * 60 * 1000)
        );
        
        let formattedUrl = apiUrl;
        
        if (!formattedUrl.startsWith('http://') && !formattedUrl.startsWith('https://')) {
            formattedUrl = 'http://' + formattedUrl;
        }
        
        if (!formattedUrl.endsWith('/')) {
            formattedUrl = formattedUrl + '/';
        }
        
        console.log('Final formatted URL being fetched:', formattedUrl);
        
        const fetchPromise = fetch(formattedUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody),
            mode: 'cors',
            credentials: 'omit'
        });
        
        const response = await Promise.race([fetchPromise, timeoutPromise]);
        
        if (!response.ok) {
            const errorText = await response.text().catch(() => '');
            throw new Error(`API error: ${response.status} - ${errorText || response.statusText}`);
        }
        
        const result = await response.json();
        
        if (debugLogging) {
            console.group('🤖 AI Response Debug');
            console.log('Response Status:', response.status);
            console.log('Response Data:', result);
            console.groupEnd();
        }
        
        hideThinking();
        addMessage(result.response, false);
        
    } catch (error) {
        console.error('Error sending message:', error);
        hideThinking();
        
        let errorMessage = "Sorry, there was an error processing your request. Please check your connection or API settings and try again.";
        
        if (error.message.includes('NetworkError') || error.message.includes('Failed to fetch') || error.message.includes('timed out')) {
            errorMessage = `Network error: Unable to connect to the API server at "${apiUrl}". Please check that the server is running and accessible.`;
        } else if (error.message.includes('CORS')) {
            errorMessage = `CORS error: The API server at "${apiUrl}" is not allowing requests from this webpage. Please ensure CORS is properly configured on the server.`;
        } else if (error.message.includes('API error')) {
            errorMessage = `Server error: ${error.message}. Please check your API server configuration.`;
        }
        
        const chatContainer = document.getElementById('chat-container');
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message assistant';
        errorDiv.innerHTML = `
            <div class="message-header">Me</div>
            <div class="message-content">${errorMessage}</div>
        `;
        chatContainer.appendChild(errorDiv);
    }
    
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ==================== Mobile Support ====================
function ensureLastMessageVisible() {
    const chatContainer = document.getElementById('chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    const messages = chatContainer.querySelectorAll('.message');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        const chatRect = chatContainer.getBoundingClientRect();
        const messageRect = lastMessage.getBoundingClientRect();
        
        const viewportHeight = window.visualViewport ? 
            window.visualViewport.height : 
            window.innerHeight;
        
        const inputTop = viewportHeight - 100;
        
        if (messageRect.bottom > inputTop) {
            const scrollOffset = messageRect.bottom - inputTop + 30;
            chatContainer.scrollTop = Math.max(0, chatContainer.scrollTop + scrollOffset);
        }
    }
}

// ==================== Tab Management ====================
function switchTab(tabName) {
    currentTab = tabName;
    
    // Update tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeButton = document.querySelector(`[data-tab="${tabName}"]`);
    activeButton.classList.add('active');
    
    // Scroll active tab into view on mobile
    if (window.innerWidth <= 768) {
        activeButton.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
        // Update opacity after scroll
        setTimeout(() => {
            const tabButtons = document.querySelector('.tab-buttons');
            if (tabButtons) {
                tabButtons.dispatchEvent(new Event('scroll'));
            }
        }, 300);
    }
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    if (tabName === 'chat') {
        document.getElementById('chat-content').classList.add('active');
        updateConnectionStatus();

        // Clean up terminal monitoring when leaving terminal tab
        if (frameValidationInterval) {
            clearInterval(frameValidationInterval);
            frameValidationInterval = null;
        }
    } else if (tabName === 'terminal') {
        document.getElementById('terminal-content').classList.add('active');
        dashboardScale = null;

        // Ensure terminal display is ready
        const terminalDisplay = document.getElementById('terminal-display');
        if (terminalDisplay && !terminalDisplay.hasChildNodes()) {
            terminalDisplay.innerHTML = '<div class="terminal-line">Connecting to terminal...</div>';
        }

        if (!terminalConnected) {
            connectTerminal();
        } else {
            // If already connected, start capture immediately
            startTerminalCapture();
        }

        updateConnectionStatus();

        // Start frame validation monitoring
        startFrameValidation();

        setTimeout(() => {
            if (terminalConnected) {
                startTerminalCapture();
            }
            setTimeout(() => {
                recalculateDashboardScale();
            }, 200);
        }, 100);
    } else if (tabName === 'spec') {
        document.getElementById('spec-content').classList.add('active');

        // Clean up terminal monitoring when leaving terminal tab
        if (frameValidationInterval) {
            clearInterval(frameValidationInterval);
            frameValidationInterval = null;
        }
        loadSpec();
        updateConnectionStatus();
    } else if (tabName === 'agents') {
        document.getElementById('agents-content').classList.add('active');

        // Clean up terminal monitoring when leaving terminal tab
        if (frameValidationInterval) {
            clearInterval(frameValidationInterval);
            frameValidationInterval = null;
        }
        if (!agentsLoaded) {
            loadAgents();
        }
        updateConnectionStatus();
    }
}

// ==================== Terminal Functions ====================
function updateConnectionStatus() {
    const terminalStatus = document.getElementById('terminal-status');
    const statusIndicator = document.getElementById('status-indicator');
    
    if (currentTab === 'terminal' && terminalConnected) {
        terminalStatus.textContent = 'Connected';
        statusIndicator.classList.add('connected');
    } else {
        terminalStatus.textContent = 'Disconnected';
        statusIndicator.classList.remove('connected');
    }
}

function connectTerminal() {
    if (terminalSocket && terminalSocket.connected) {
        return;
    }

    // Clear any pending reconnection attempts
    if (terminalReconnectTimeout) {
        clearTimeout(terminalReconnectTimeout);
        terminalReconnectTimeout = null;
    }

    console.log('Connecting to terminal WebSocket (attempt', terminalReconnectAttempts + 1, ')');
    
    const pathname = window.location.pathname;
    let socketPath = '/socket.io';
    
    if (pathname && pathname !== '/') {
        const cleanPath = pathname.replace(/\/$/, '');
        socketPath = cleanPath + '/socket.io';
    }
    
    console.log('Using socket.io path:', socketPath);
    
    terminalSocket = io.connect('/terminal', {
        path: socketPath,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: 5
    });
    
    terminalSocket.on('connect', () => {
        console.log('Terminal WebSocket connected');
        terminalConnected = true;
        terminalReconnectAttempts = 0;
        frameStuckCounter = 0;
        lastFrameUpdateTime = Date.now();
        updateConnectionStatus();

        // Request initial frame after successful connection
        setTimeout(() => {
            if (terminalSocket && terminalConnected && currentTab === 'terminal') {
                startTerminalCapture();
            }
        }, 100);
    });
    
    terminalSocket.on('disconnect', () => {
        console.log('Terminal WebSocket disconnected');
        terminalConnected = false;
        updateConnectionStatus();

        // Attempt reconnection if we're still on the terminal tab
        if (currentTab === 'terminal' && terminalReconnectAttempts < 5) {
            terminalReconnectAttempts++;
            terminalReconnectTimeout = setTimeout(() => {
                console.log('Attempting to reconnect terminal...');
                connectTerminal();
            }, Math.min(1000 * Math.pow(2, terminalReconnectAttempts), 10000));
        }
    });
    
    terminalSocket.on('connect_error', (error) => {
        console.error('Terminal WebSocket connection error:', error);

        // If connection fails and we're on terminal tab, show error message
        if (currentTab === 'terminal') {
            const terminalDisplay = document.getElementById('terminal-display');
            if (terminalDisplay && terminalDisplay.children.length === 0) {
                terminalDisplay.innerHTML = '<div class="terminal-line">Connection error. Retrying...</div>';
            }
        }
    });
    
    terminalSocket.on('terminal_output', (data) => {
        appendTerminalOutput(data.data);
    });
    
    // Handle new differential updates with validation
    terminalSocket.on('dashboard_update', (data) => {
        // Debug: Log update type and size
        if (window.dashboardDebug) {
            const size = JSON.stringify(data).length;
            console.log(`Dashboard update: type=${data.type}, size=${size} bytes, changes=${data.changes ? data.changes.length : 0}`);
        }

        // Compute frame hash for stuck detection
        const frameContent = JSON.stringify(data.frame || data.changes || []);
        const currentHash = simpleHash(frameContent);

        // Check if frame is stuck (same content repeatedly)
        if (currentHash === lastFrameHash) {
            frameStuckCounter++;

            // If frame has been stuck for too long, force a refresh
            if (frameStuckCounter > 50) {
                console.log('Frame appears stuck, requesting full refresh');
                frameStuckCounter = 0;
                lastFrameHash = null;

                // Request fresh frame
                if (terminalSocket && terminalConnected) {
                    forceTerminalRefresh();
                }
            }
        } else {
            frameStuckCounter = 0;
            lastFrameHash = currentHash;
            lastFrameUpdateTime = Date.now();
        }

        if (data.type === 'full') {
            // Full frame update - render entire dashboard
            if (data.frame && Array.isArray(data.frame)) {
                renderDashboardFrame(data.frame);
            }
        } else if (data.type === 'diff') {
            // Differential update - apply only changes
            applyDashboardDiff(data.changes);
        }
    });
    
    terminalSocket.on('terminal_init', (data) => {
        if (data.lines) {
            const terminalDisplay = document.getElementById('terminal-display');
            terminalDisplay.innerHTML = '';
            data.lines.forEach(line => {
                appendTerminalOutput(line);
            });
        }
    });
    
    terminalSocket.on('capture_started', (data) => {
        if (data.status === 'connected_to_existing') {
            document.getElementById('terminal-display').innerHTML = '';
        } else if (data.status === 'no_dashboard_found') {
            appendTerminalOutput('No active dashboard found. Start training to see dashboard output.');
        }
    });
    
    terminalSocket.on('capture_stopped', () => {
        appendTerminalOutput('Dashboard connection stopped.');
    });
}

function appendTerminalOutput(text) {
    const terminalDisplay = document.getElementById('terminal-display');
    const lineDiv = document.createElement('div');
    lineDiv.className = 'terminal-line';
    lineDiv.textContent = text;
    terminalDisplay.appendChild(lineDiv);
    
    while (terminalDisplay.children.length > 1000) {
        terminalDisplay.removeChild(terminalDisplay.firstChild);
    }
    
    terminalDisplay.scrollTop = terminalDisplay.scrollHeight;
}

function startTerminalCapture() {
    if (terminalSocket && terminalConnected) {
        terminalSocket.emit('start_capture', {
            command: 'connect_existing'
        });
    }
}

// Helper function to compute simple hash for frame comparison
function simpleHash(str) {
    let hash = 0;
    if (str.length === 0) return hash;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32bit integer
    }
    return hash;
}

// Force terminal refresh when stuck
function forceTerminalRefresh() {
    console.log('Forcing terminal refresh...');
    const terminalDisplay = document.getElementById('terminal-display');
    if (terminalDisplay) {
        terminalDisplay.innerHTML = '<div class="terminal-line">Refreshing display...</div>';
    }

    // Stop and restart capture
    if (terminalSocket && terminalConnected) {
        terminalSocket.emit('stop_capture');
        setTimeout(() => {
            startTerminalCapture();
        }, 500);
    }
}

// Start frame validation monitoring
function startFrameValidation() {
    // Clear any existing interval
    if (frameValidationInterval) {
        clearInterval(frameValidationInterval);
    }

    // Monitor frame updates every 5 seconds
    frameValidationInterval = setInterval(() => {
        if (currentTab !== 'terminal') {
            clearInterval(frameValidationInterval);
            frameValidationInterval = null;
            return;
        }

        // Check if we haven't received updates in a while
        const timeSinceLastUpdate = Date.now() - lastFrameUpdateTime;
        if (timeSinceLastUpdate > 10000 && terminalConnected) {
            console.log('No frame updates received for 10s, refreshing...');
            forceTerminalRefresh();
            lastFrameUpdateTime = Date.now(); // Reset to prevent rapid refreshes
        }
    }, 5000);
}

// Validate terminal rendering state
function validateTerminalRendering() {
    const terminalDisplay = document.getElementById('terminal-display');
    if (!terminalDisplay) return;

    // Check if display is empty when it shouldn't be
    if (terminalConnected && terminalDisplay.children.length === 0) {
        console.log('Terminal display is empty, initializing...');
        terminalDisplay.innerHTML = '<div class="terminal-line">Initializing terminal display...</div>';
        startTerminalCapture();
    }

    // Check if frame container exists but is not visible
    if (currentFrameContainer && currentFrameContainer.offsetHeight === 0) {
        console.log('Frame container is hidden, recalculating scale...');
        recalculateDashboardScale();
    }
}

// Dashboard frame buffer for differential updates
let dashboardFrameBuffer = [];
let dashboardInitialized = false;

function renderDashboardFrame(frame) {
    const terminalDisplay = document.getElementById('terminal-display');
    terminalDisplay.innerHTML = '';
    
    // Store frame in buffer for differential updates
    dashboardFrameBuffer = frame.slice();
    dashboardInitialized = true;
    
    const wrapperDiv = document.createElement('div');
    wrapperDiv.style.position = 'relative';
    wrapperDiv.style.overflow = 'hidden';
    wrapperDiv.style.backgroundColor = '#0d0d0d';
    wrapperDiv.style.display = 'block';
    wrapperDiv.style.margin = '0 auto';
    wrapperDiv.style.userSelect = 'text';  // Enable text selection
    wrapperDiv.style.cursor = 'text';      // Show text cursor
    
    const frameContainer = document.createElement('div');
    frameContainer.className = 'dashboard-frame';
    frameContainer.style.userSelect = 'text';  // Enable text selection
    
    const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    frame.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'dashboard-line';
        lineDiv.setAttribute('data-line-index', index);
        lineDiv.style.userSelect = 'text';  // Enable text selection per line
        
        if (isMobileDevice) {
            // Replace Unicode box drawing characters with ASCII on mobile
            line = line.replace(/[█▓▒░]/g, '#')
                      .replace(/[▀▄]/g, '=')
                      .replace(/[▌▐]/g, '|')
                      .replace(/[■□▪▫◼◻◾◽▬▭▮▯]/g, '#')
                      .replace(/[═]/g, '=')
                      .replace(/[─━╌╍┄┅┈┉⎯⎼⎽]/g, '-')
                      .replace(/[▁▂▃▄▅▆▇]/g, '_')
                      .replace(/[⌐¬]/g, '-')
                      .replace(/[·•◦]/g, '*')
                      .replace(/[◯○]/g, 'o')
                      .replace(/[●◉]/g, '*')
                      .replace(/[║┃│|╎╏┆┇┊┋]/g, '|')
                      .replace(/[┏┌┍┎╔╒╓╭┓┐┑┒╗╕╖╮┗└┕┖╚╘╙╰┛┘┙┚╝╛╜╯]/g, '+')
                      .replace(/[┣├┝┞┟┠┡┢╟╞╠┫┤┥┦┧┨┩┪╢╡╣]/g, '+')
                      .replace(/[┳┬┭┮┯┰┱┲╦╤╥┻┴┵┶┷┸┹┺╩╧╨]/g, '+')
                      .replace(/[╋┼┽┾┿╀╁╂╬╪╫]/g, '+');
        }
        
        lineDiv.textContent = line;
        frameContainer.appendChild(lineDiv);
    });
    
    wrapperDiv.appendChild(frameContainer);
    terminalDisplay.appendChild(wrapperDiv);
    
    currentFrameContainer = frameContainer;
    currentWrapperDiv = wrapperDiv;
    
    if (dashboardScale === null) {
        requestAnimationFrame(() => {
            calculateDashboardScale();
        });
    } else if (dashboardScale) {
        applyDashboardScale();
    }
}

function applyDashboardDiff(changes) {
    if (!dashboardInitialized || !currentFrameContainer) {
        // No frame to update yet, wait for full frame
        return;
    }

    const isMobileDevice = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

    // Track if we're updating important regions and if changes are actually happening
    let hasImportantUpdates = false;
    let actualChanges = 0;

    // Apply each change
    changes.forEach(change => {
        const { row, col, text, length } = change;

        // Update buffer
        if (row < dashboardFrameBuffer.length) {
            const line = dashboardFrameBuffer[row];
            const before = line.substring(0, col);
            const after = line.substring(col + length);
            const oldContent = line.substring(col, col + length);

            // Check if this is an important region (HASH/CONTEXT sections)
            if (line.includes('HASH') || line.includes('CONTEXT') ||
                line.includes('TOKEN') || line.includes('STEP')) {
                hasImportantUpdates = true;
            }

            // Only update if content actually changed
            if (oldContent !== text) {
                actualChanges++;
                dashboardFrameBuffer[row] = before + text + after;

                // Update DOM
                const lineDiv = currentFrameContainer.querySelector(`[data-line-index="${row}"]`);
            if (lineDiv) {
                let updatedLine = dashboardFrameBuffer[row];
                
                if (isMobileDevice) {
                    // Apply mobile character replacements
                    updatedLine = updatedLine.replace(/[█▓▒░]/g, '#')
                              .replace(/[▀▄]/g, '=')
                              .replace(/[▌▐]/g, '|')
                              .replace(/[■□▪▫◼◻◾◽▬▭▮▯]/g, '#')
                              .replace(/[═]/g, '=')
                              .replace(/[─━╌╍┄┅┈┉⎯⎼⎽]/g, '-')
                              .replace(/[▁▂▃▄▅▆▇]/g, '_')
                              .replace(/[⌐¬]/g, '-')
                              .replace(/[·•◦]/g, '*')
                              .replace(/[◯○]/g, 'o')
                              .replace(/[●◉]/g, '*')
                              .replace(/[║┃│|╎╏┆┇┊┋]/g, '|')
                              .replace(/[┏┌┍┎╔╒╓╭┓┐┑┒╗╕╖╮┗└┕┖╚╘╙╰┛┘┙┚╝╛╜╯]/g, '+')
                              .replace(/[┣├┝┞┟┠┡┢╟╞╠┫┤┥┦┧┨┩┪╢╡╣]/g, '+')
                              .replace(/[┳┬┭┮┯┰┱┲╦╤╥┻┴┵┶┷┸┹┺╩╧╨]/g, '+')
                              .replace(/[╋┼┽┾┿╀╁╂╬╪╫]/g, '+');
                }
                
                // Only update if text actually changed to preserve selection
                if (lineDiv.textContent !== updatedLine) {
                    // Save any existing text selection
                    const selection = window.getSelection();
                    const hasSelection = selection.rangeCount > 0 && !selection.isCollapsed;
                    let savedRange = null;
                    
                    // Check if selection intersects with this line
                    if (hasSelection) {
                        const range = selection.getRangeAt(0);
                        if (lineDiv.contains(range.commonAncestorContainer)) {
                            // Save selection relative to line start
                            savedRange = {
                                startOffset: range.startOffset,
                                endOffset: range.endOffset
                            };
                        }
                    }
                    
                    // Update the text content
                    lineDiv.textContent = updatedLine;
                    
                    // Restore selection if it was in this line
                    if (savedRange && lineDiv.firstChild) {
                        try {
                            const newRange = document.createRange();
                            const textNode = lineDiv.firstChild;
                            const maxOffset = textNode.textContent.length;
                            
                            // Clamp offsets to valid range
                            const startOffset = Math.min(savedRange.startOffset, maxOffset);
                            const endOffset = Math.min(savedRange.endOffset, maxOffset);
                            
                            newRange.setStart(textNode, startOffset);
                            newRange.setEnd(textNode, endOffset);
                            
                            selection.removeAllRanges();
                            selection.addRange(newRange);
                        } catch (e) {
                            // Selection restoration failed, ignore
                        }
                    }
                }
            }
            }
        }
    });

    // If we had important updates but no actual changes, the display might be stuck
    if (hasImportantUpdates && actualChanges === 0) {
        frameStuckCounter++;
        if (frameStuckCounter > 20) {
            console.log('Important regions not updating despite changes, forcing refresh...');
            forceTerminalRefresh();
            frameStuckCounter = 0;
        }
    } else if (actualChanges > 0) {
        // Reset counter when we have actual changes
        frameStuckCounter = 0;
    }
}

function calculateDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv) return;
    
    const terminalDisplay = document.getElementById('terminal-display');
    const naturalWidth = currentFrameContainer.scrollWidth;
    const naturalHeight = currentFrameContainer.scrollHeight;
    
    currentFrameContainer.style.width = naturalWidth + 'px';
    currentFrameContainer.style.height = naturalHeight + 'px';
    currentFrameContainer.style.maxWidth = naturalWidth + 'px';
    currentFrameContainer.style.overflow = 'hidden';
    
    const containerWidth = terminalDisplay.clientWidth;
    const containerHeight = terminalDisplay.clientHeight || window.innerHeight * 0.6;
    const padding = 20;
    
    const widthScale = (containerWidth - padding) / naturalWidth;
    const heightScale = (containerHeight - padding) / naturalHeight;
    
    dashboardScale = Math.min(widthScale, heightScale, 1.5);
    
    if (window.innerWidth <= 768 && dashboardScale > 1) {
        dashboardScale = 1;
    }
    
    applyDashboardScale();
}

function applyDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv || !dashboardScale) return;
    
    const terminalDisplay = document.getElementById('terminal-display');
    const naturalWidth = currentFrameContainer.scrollWidth || parseInt(currentFrameContainer.style.width);
    const naturalHeight = currentFrameContainer.scrollHeight || parseInt(currentFrameContainer.style.height);
    
    currentFrameContainer.style.transform = `scale(${dashboardScale})`;
    currentFrameContainer.style.transformOrigin = 'top left';
    
    const scaledHeight = naturalHeight * dashboardScale;
    const scaledWidth = naturalWidth * dashboardScale;
    
    currentWrapperDiv.style.width = scaledWidth + 'px';
    currentWrapperDiv.style.height = scaledHeight + 'px';
    currentWrapperDiv.style.overflow = 'hidden';
    currentWrapperDiv.style.margin = '0 auto';
    currentWrapperDiv.style.display = 'block';
    
    terminalDisplay.style.minHeight = (scaledHeight + 20) + 'px';
    terminalDisplay.scrollTop = 0;
}

function recalculateDashboardScale() {
    if (!currentFrameContainer || !currentWrapperDiv || currentTab !== 'terminal') {
        return;
    }
    
    dashboardScale = null;
    currentFrameContainer.style.transform = 'none';
    calculateDashboardScale();
}

// ==================== Spec Tab Functions ====================
async function loadSpec() {
    if (specLoaded) return;
    
    const container = document.getElementById('spec-container');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/spec`);
        if (!response.ok) throw new Error('Failed to fetch spec');
        
        const data = await response.json();
        specLoaded = true;
        
        // Update logo icon with seed value if available
        if (data.seed) {
            const logoIcon = document.querySelector('.logo-icon');
            if (logoIcon) {
                logoIcon.setAttribute('data-seed', data.seed);
            }
        }
        
        let html = '';
        
        // Identity section with hashes
        if (data.full_hash && data.truncated_hash) {
            html += '<div class="spec-section">';
            html += '<div class="spec-title">Hashes</div>';
            
            const truncLen = data.truncated_hash.length;
            const truncPart = data.full_hash.substring(0, truncLen);
            const restPart = data.full_hash.substring(truncLen);
            html += '<div class="spec-hash">';
            html += `<a href="#args-title" style="color: #4caf50; font-weight: 600; text-decoration: none;">${truncPart}</a>`;
            html += `<span style="color: var(--text);">${restPart}</span>`;
            html += '</div>';
            html += '</div>';
        }
        
        // Checkout section
        if (data.git_url) {
            html += '<div class="spec-section">';
            html += '<div class="spec-title">Commands</div>';
            html += '<div class="spec-code-block">';
            html += 'Clone it from the source:'
            html += `<div class="spec-metadata"><code style="background: #f5f5f5; color: #333; padding: 2px 4px; border-radius: 3px; font-family: 'Cascadia Code', 'Fira Code', monospace;">git clone ${data.git_url}</code></div>`;
            html += `Move into the directory:`
            html += `<div class="spec-metadata"><code style="background: #f5f5f5; color: #333; padding: 2px 4px; border-radius: 3px; font-family: 'Cascadia Code', 'Fira Code', monospace;">cd praxis</code></div>`;
            html += `Reproduce the experiment:`
            html += `<div class="spec-metadata"><code style="background: #f5f5f5; color: #333; padding: 2px 4px; border-radius: 3px; font-family: 'Cascadia Code', 'Fira Code', monospace;">${data.command.replace('python main.py', './launch')}</code></div>`;
            html += '</div>';
            html += '</div>';
        }
        
        // Parameter statistics (simplified)
        if (data.param_stats) {
            html += '<div class="spec-section">';
            html += '<div class="spec-title">Parameters</div>';
            
            if (data.param_stats.model_parameters) {
                html += `<div class="spec-metadata">Model Parameters: <span style="color: #4caf50; font-weight: 600;">${data.param_stats.model_parameters.toLocaleString()}</span></div>`;
            }
            
            if (data.param_stats.optimizer_parameters) {
                html += `<div class="spec-metadata">Optimizer Parameters: <span style="color: #4caf50; font-weight: 600;">${data.param_stats.optimizer_parameters.toLocaleString()}</span></div>`;
            }
            
            html += '</div>';
        }
        
        // Model architecture
        if (data.model_architecture) {
            html += '<div class="spec-section">';
            html += '<div class="spec-title">Specification</div>';
            html += `<pre class="spec-code">${data.model_architecture}</pre>`;
            html += '</div>';
        }
        
        html += '<div id="args-title" class="args-title">Arguments</div>';
        
        if (data.timestamp) {
            html += `<div class="spec-metadata">Created: ${data.timestamp}</div>`;
        }
        
        if (data.args) {
            html += `<pre class="spec-code">${JSON.stringify(data.args, null, 2)}</pre>`;
        }
        
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading spec:', error);
        container.innerHTML = '<div style="padding: 20px; color: var(--light-text);">Error loading model specification</div>';
    }
}

// ==================== Agents Tab Functions ====================
async function loadAgents() {
    const container = document.getElementById('agents-container');
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/agents`);
        const data = await response.json();
        
        if (data.error) {
            container.innerHTML = `<div class="agents-error">Error: ${data.error}</div>`;
            return;
        }
        
        // All agents (including self instances) now come from the backend
        if (!data.agents || data.agents.length === 0) {
            container.innerHTML = '<div class="agents-empty">No agents found. Add remotes using: git remote add &lt;name&gt; &lt;url&gt;</div>';
            return;
        }
        
        // Check for duplicate masked URLs
        const maskedUrlCounts = {};
        data.agents.forEach(agent => {
            const maskedUrl = agent.masked_url || agent.url;
            maskedUrlCounts[maskedUrl] = (maskedUrlCounts[maskedUrl] || 0) + 1;
        });
        
        // Build the agents section
        let html = '<div class="agents-section">';
        html += '<div class="agents-title">Remotes</div>';
        html += '<div class="agents-table">';
        html += '<div class="agents-list">';
        
        data.agents.forEach((agent) => {
            const maskedUrl = agent.masked_url || agent.url;
            const isDuplicate = maskedUrlCounts[maskedUrl] > 1;
            
            let statusClass, statusText;
            if (isDuplicate) {
                statusClass = 'ambiguous';
                statusText = 'Ambiguous';
            } else if (agent.status === 'online') {
                statusClass = 'online';
                statusText = 'Online';
            } else if (agent.status === 'archived') {
                statusClass = 'archived';
                statusText = 'Archived';
            } else {
                statusClass = 'offline';
                statusText = 'Unknown';
            }
            
            let animationStyle = '';
            if (statusClass === 'online') {
                const duration = (1.5 + Math.random() * 2).toFixed(2);
                const delay = (Math.random() * 2).toFixed(2);
                animationStyle = `style="animation-duration: ${duration}s; animation-delay: ${delay}s;"`;
            }
            
            html += `
                <div class="agent-row">
                    <div class="agent-info">
                        <div class="agent-name">${agent.name}</div>
                        <div class="agent-url">${agent.masked_url || agent.url}${agent.short_hash ? ` | ${agent.short_hash}` : ''}</div>
                    </div>
                    <div class="agent-status ${statusClass}">
                        <span class="status-dot ${statusClass}" ${animationStyle}></span>
                        ${statusText}
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        html += '</div>';
        html += '</div>';
        
        container.innerHTML = html;
        agentsLoaded = true;
        
    } catch (error) {
        container.innerHTML = `<div class="agents-error">Failed to load agents: ${error.message}</div>`;
    }
}

// ==================== Event Listeners Setup ====================
// ==================== Mobile Tab Carousel ====================
function setupTabCarousel() {
    const tabButtons = document.querySelector('.tab-buttons');
    const tabNav = document.querySelector('.tab-nav');
    if (!tabButtons || !tabNav || window.innerWidth > 768) return;
    
    let scrollEndTimer = null;
    let isScrolling = false;
    let userInitiatedScroll = false;
    
    function updateTabState() {
        const buttons = tabButtons.querySelectorAll('.tab-button');
        const containerRect = tabButtons.getBoundingClientRect();
        const scrollLeft = tabButtons.scrollLeft;
        const scrollWidth = tabButtons.scrollWidth;
        const clientWidth = tabButtons.clientWidth;
        
        // Get terminal-status container position for right boundary
        const terminalStatus = document.querySelector('.terminal-status');
        const terminalRect = terminalStatus ? terminalStatus.getBoundingClientRect() : null;
        const rightBoundary = terminalRect ? terminalRect.left - 10 : containerRect.right - 120;
        
        // Update scroll indicators
        if (scrollLeft > 5) {
            tabNav.classList.add('has-scroll-left');
        } else {
            tabNav.classList.remove('has-scroll-left');
        }
        
        if (scrollLeft < scrollWidth - clientWidth - 5) {
            tabNav.classList.add('has-scroll-right');
        } else {
            tabNav.classList.remove('has-scroll-right');
        }
        
        // Apply compression effect as buttons approach right boundary
        let cumulativeOffset = 0;
        buttons.forEach((button, index) => {
            const rect = button.getBoundingClientRect();
            const buttonLeft = rect.left;
            const distanceFromRight = rightBoundary - buttonLeft;
            
            if (distanceFromRight < 300 && distanceFromRight > -150) {
                // Even wider compression zone
                const normalizedDistance = Math.max(0, Math.min(1, (distanceFromRight + 150) / 450));
                
                // Extremely aggressive exponential compression
                const compressionAmount = Math.pow(1 - normalizedDistance, 4);
                
                // VERY strong stacking - buttons almost completely overlap
                const baseCompression = compressionAmount * 200; // Massive base compression
                const stackingFactor = compressionAmount * (index * index * 2); // Much stronger exponential stacking
                
                // Add to cumulative offset for extreme cascading
                cumulativeOffset += (baseCompression + stackingFactor) * 0.8;
                
                // Calculate final position - buttons heavily slide left and stack
                const translateX = -cumulativeOffset;
                
                // Dramatic scale and opacity changes for deck effect
                const scale = 1 - (compressionAmount * 0.25); // Scale down to 75% when fully compressed
                const opacity = Math.max(0.2, 1 - (compressionAmount * 0.7)); // Fade to 20% opacity minimum
                
                // Z-index creates proper stacking order
                const zIndex = 100 - index; // Later buttons go under
                
                button.style.transform = `translateX(${translateX}px) scale(${scale})`;
                button.style.opacity = opacity;
                button.style.zIndex = zIndex;
                button.style.transition = 'transform 0.1s ease-out, opacity 0.1s ease-out';
                
                // Active button should always be visible but still affected
                if (button.classList.contains('active')) {
                    button.style.opacity = Math.max(0.85, opacity);
                    button.style.transform = `translateX(${translateX}px) scale(${Math.max(0.9, scale)})`;
                    button.style.zIndex = '200'; // Active button always on top
                }
            } else {
                // Reset transformation for buttons outside compression zone
                button.style.transform = '';
                button.style.opacity = '';
                button.style.zIndex = '';
            }
        });
        
        // Ensure active button is always visible
        const activeButton = tabButtons.querySelector('.tab-button.active');
        if (activeButton && scrollLeft === 0) {
            // On initial load, make sure active button is visible
            const activeRect = activeButton.getBoundingClientRect();
            
            if (activeRect.left < containerRect.left + 30 || 
                activeRect.right > containerRect.right - 30) {
                // Active button is in fade zone, scroll it into view
                activeButton.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            }
        }
    }
    
    function snapToNearestButton() {
        if (isScrolling || !userInitiatedScroll) return;
        
        const buttons = tabButtons.querySelectorAll('.tab-button');
        const containerRect = tabButtons.getBoundingClientRect();
        // Align to where the first button normally sits (left edge + padding)
        const snapTarget = containerRect.left + 40; // 40px is the padding-left
        
        let closestButton = null;
        let closestDistance = Infinity;
        
        buttons.forEach(button => {
            const rect = button.getBoundingClientRect();
            const distance = Math.abs(rect.left - snapTarget);
            
            if (distance < closestDistance) {
                closestDistance = distance;
                closestButton = button;
            }
        });
        
        if (closestButton && closestDistance > 15) {
            const rect = closestButton.getBoundingClientRect();
            const offset = rect.left - snapTarget;
            
            // Mark that we're doing a programmatic scroll
            userInitiatedScroll = false;
            
            // Use native smooth scrolling
            tabButtons.scrollBy({
                left: offset,
                behavior: 'smooth'
            });
        }
    }
    
    // Add smooth scrolling behavior to the container
    tabButtons.style.scrollBehavior = 'smooth';
    
    // Update on scroll
    tabButtons.addEventListener('scroll', () => {
        requestAnimationFrame(updateTabState);
        
        isScrolling = true;
        
        // Clear existing timer
        if (scrollEndTimer) {
            clearTimeout(scrollEndTimer);
        }
        
        // Detect when scrolling stops
        scrollEndTimer = setTimeout(() => {
            isScrolling = false;
            // Only snap if this was a user-initiated scroll
            if (userInitiatedScroll) {
                snapToNearestButton();
            }
        }, 150);
    });
    
    // Add momentum on touch devices
    let touchStartX = 0;
    let touchStartTime = 0;
    let lastTouchX = 0;
    let lastTouchTime = 0;
    
    tabButtons.addEventListener('touchstart', (e) => {
        touchStartX = e.touches[0].clientX;
        touchStartTime = Date.now();
        lastTouchX = touchStartX;
        lastTouchTime = touchStartTime;
        userInitiatedScroll = true; // Mark as user-initiated
    }, { passive: true });
    
    tabButtons.addEventListener('touchmove', (e) => {
        lastTouchX = e.touches[0].clientX;
        lastTouchTime = Date.now();
        userInitiatedScroll = true; // Keep marking as user-initiated
    }, { passive: true });
    
    tabButtons.addEventListener('touchend', () => {
        const touchEndTime = Date.now();
        const timeDiff = touchEndTime - lastTouchTime;
        
        // If the touch ended very recently after last move, we have momentum
        if (timeDiff < 50) {
            const velocity = (lastTouchX - touchStartX) / (touchEndTime - touchStartTime);
            
            // Add a small momentum boost in the direction of swipe
            if (Math.abs(velocity) > 0.3) {
                const boost = velocity * 100;
                tabButtons.scrollBy({
                    left: -boost,
                    behavior: 'smooth'
                });
            }
        }
    }, { passive: true });
    
    // Also track wheel scrolling as user-initiated
    tabButtons.addEventListener('wheel', () => {
        userInitiatedScroll = true;
    }, { passive: true });
    
    // Add mouse drag support for desktop users
    let isDragging = false;
    let mouseStartX = 0;
    let scrollStartX = 0;
    let mouseLastX = 0;
    let mouseStartTime = 0;
    
    tabButtons.addEventListener('mousedown', (e) => {
        isDragging = true;
        mouseStartX = e.clientX;
        mouseLastX = e.clientX;
        scrollStartX = tabButtons.scrollLeft;
        mouseStartTime = Date.now();
        userInitiatedScroll = true;
        
        // Prevent text selection while dragging
        e.preventDefault();
        
        // Change cursor to grabbing
        tabButtons.style.cursor = 'grabbing';
        tabButtons.style.userSelect = 'none';
    });
    
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        
        e.preventDefault();
        const deltaX = e.clientX - mouseStartX;
        tabButtons.scrollLeft = scrollStartX - deltaX;
        mouseLastX = e.clientX;
        userInitiatedScroll = true;
    });
    
    window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        
        isDragging = false;
        
        // Reset cursor
        tabButtons.style.cursor = '';
        tabButtons.style.userSelect = '';
        
        // Calculate velocity for momentum
        const mouseEndTime = Date.now();
        const timeDiff = mouseEndTime - mouseStartTime;
        
        if (timeDiff > 0 && timeDiff < 200) {
            const velocity = (mouseLastX - mouseStartX) / timeDiff;
            
            // Add momentum boost for quick drags
            if (Math.abs(velocity) > 0.3) {
                const boost = velocity * 100;
                tabButtons.scrollBy({
                    left: -boost,
                    behavior: 'smooth'
                });
            }
        }
    });
    
    // Handle mouse leave to cancel dragging if cursor leaves window
    window.addEventListener('mouseleave', () => {
        if (isDragging) {
            isDragging = false;
            tabButtons.style.cursor = '';
            tabButtons.style.userSelect = '';
        }
    });
    
    // Initial update
    setTimeout(updateTabState, 100);
    
    // Check for initial overflow
    if (tabButtons.scrollWidth > tabButtons.clientWidth) {
        tabNav.classList.add('has-scroll-right');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    // Setup live reload
    setupLiveReload();
    
    // Setup mobile tab carousel
    setupTabCarousel();
    
    // Fetch and display seed value immediately
    fetch(`${API_BASE_URL}/api/spec`)
        .then(response => response.json())
        .then(data => {
            if (data.seed) {
                const logoIcon = document.querySelector('.logo-icon');
                if (logoIcon) {
                    logoIcon.setAttribute('data-seed', data.seed);
                }
            }
        })
        .catch(err => console.error('Failed to fetch seed:', err));
    
    // Get DOM elements
    const messageInput = document.getElementById('message-input');
    const settingsButton = document.getElementById('settings-button');
    const themeToggleButton = document.getElementById('theme-toggle');
    const closeModalButton = document.getElementById('close-modal');
    const saveSettingsButton = document.getElementById('save-settings');
    const resetButton = document.getElementById('reset-settings');
    const developerPromptElement = document.getElementById('developer-prompt');
    const settingsModal = document.getElementById('settings-modal');
    const temperatureInput = document.getElementById('temperature');
    const temperatureValue = document.getElementById('temperature-value');
    const repetitionPenaltyInput = document.getElementById('repetition-penalty');
    const repetitionPenaltyValue = document.getElementById('repetition-penalty-value');
    
    // Tab elements
    const chatTab = document.getElementById('chat-tab');
    const terminalTab = document.getElementById('terminal-tab');
    const specTab = document.getElementById('spec-tab');
    const agentsTab = document.getElementById('agents-tab');
    
    // Initialize message input
    messageInput.value = PREFIX + PLACEHOLDER_TEXT;
    messageInput.style.color = 'var(--light-text)';
    messageInput.style.fontStyle = 'italic';
    setCursorAfterPrefix();
    
    // Load saved settings
    loadSettings();
    
    // Message input event listeners
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
            return;
        }
        
        const cursorPos = messageInput.selectionStart;
        if ((e.key === 'ArrowLeft' || e.key === 'Home' || e.key === 'Backspace') && cursorPos <= PREFIX.length) {
            if (e.key === 'Backspace' && cursorPos === PREFIX.length) {
                e.preventDefault();
            } else if (e.key === 'ArrowLeft' && cursorPos === PREFIX.length) {
                e.preventDefault();
            } else if (e.key === 'Home') {
                e.preventDefault();
                setCursorAfterPrefix();
            }
        }
    });
    
    messageInput.addEventListener('input', maintainPrefix);
    
    messageInput.addEventListener('focus', () => {
        hidePlaceholder();
        // Only set cursor position if the input only contains the prefix (no user text)
        if (messageInput.value === PREFIX) {
            setTimeout(setCursorAfterPrefix, 0);
        }
        ensureLastMessageVisible();
        setTimeout(ensureLastMessageVisible, 350);
        setTimeout(ensureLastMessageVisible, 600);
    });
    
    messageInput.addEventListener('blur', () => {
        showPlaceholder();
    });
    
    messageInput.addEventListener('click', () => {
        if (isShowingPlaceholder) {
            hidePlaceholder();
        }
        const cursorPos = messageInput.selectionStart;
        if (cursorPos < PREFIX.length) {
            setCursorAfterPrefix();
        }
    });
    
    // Settings and theme event listeners
    themeToggleButton.addEventListener('click', toggleTheme);
    settingsButton.addEventListener('click', openModal);
    closeModalButton.addEventListener('click', closeModal);
    
    saveSettingsButton.addEventListener('click', () => {
        saveSettings().catch(err => {
            console.error('Error in save settings:', err);
            const confirmationDiv = document.getElementById('save-confirmation');
            confirmationDiv.textContent = 'Error saving settings. Check console.';
            confirmationDiv.style.backgroundColor = '#d32f2f';
            confirmationDiv.classList.add('show');
            
            setTimeout(() => {
                confirmationDiv.classList.remove('show');
                confirmationDiv.style.backgroundColor = '';
            }, 3000);
        });
    });
    
    resetButton.addEventListener('click', () => {
        const keysToRemove = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && (key.startsWith('praxis_') || key === 'theme' || key === 'chatHistory')) {
                keysToRemove.push(key);
            }
        }
        
        keysToRemove.forEach(key => localStorage.removeItem(key));
        
        const confirmationDiv = document.getElementById('save-confirmation');
        confirmationDiv.textContent = 'All settings cleared! Refreshing...';
        confirmationDiv.classList.add('show');
        
        setTimeout(() => window.location.reload(), 1500);
    });
    
    // Parameter input event listeners
    temperatureInput.addEventListener('input', () => {
        temperatureValue.textContent = temperatureInput.value;
    });
    
    repetitionPenaltyInput.addEventListener('input', () => {
        repetitionPenaltyValue.textContent = repetitionPenaltyInput.value;
    });
    
    // Modal click outside to close
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            closeModal();
        }
    });
    
    // Save developer prompt on change
    developerPromptElement.addEventListener('blur', () => {
        localStorage.setItem('praxis_developer_prompt', developerPromptElement.innerHTML.trim());
    });
    
    // Tab event listeners
    chatTab.addEventListener('click', () => switchTab('chat'));
    terminalTab.addEventListener('click', () => switchTab('terminal'));
    specTab.addEventListener('click', () => {
        switchTab('spec');
        if (!specLoaded) loadSpec();
    });
    agentsTab.addEventListener('click', () => {
        switchTab('agents');
        if (!agentsLoaded) {
            loadAgents();
            if (agentsRefreshInterval) clearInterval(agentsRefreshInterval);
            agentsRefreshInterval = setInterval(() => {
                if (currentTab === 'agents') {
                    loadAgents();
                }
            }, 30000);
        }
    });
    
    // Visual Viewport API for mobile keyboard detection
    if (window.visualViewport) {
        let previousHeight = window.visualViewport.height;
        
        window.visualViewport.addEventListener('resize', () => {
            const currentHeight = window.visualViewport.height;
            
            if (currentHeight < previousHeight && document.activeElement === messageInput) {
                setTimeout(ensureLastMessageVisible, 100);
            }
            
            previousHeight = currentHeight;
        });
    }
    
    // Handle orientation changes
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            recalculateDashboardScale();
        }, 100);
    });
    
    // Handle window resize
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            recalculateDashboardScale();
        }, 250);
    });
});