        // ============= CONFIGURATION =============
        // ============= SESSION ID =============
        const session_id = localStorage.getItem('session_id') || (() => {
            const id = crypto.randomUUID();
            localStorage.setItem('session_id', id);
            return id;
        })();

        const API_URL = window.location.hostname === 'localhost' 
            ? 'http://localhost:5000/query'
            : `${window.location.origin}/query`;
        
        // ============= DOM ELEMENTS =============
        const chatContainer = document.getElementById('chatContainer');
        const queryInput = document.getElementById('queryInput');
        const sendBtn = document.getElementById('sendBtn');
        const clearBtn = document.getElementById('clearBtn');
        const emptyState = document.getElementById('emptyState');
        const statsDisplay = document.getElementById('statsDisplay');
        const progressIndicator = document.getElementById('progressIndicator');
        const progressText = document.getElementById('progressText');
        const quickActionCards = document.querySelectorAll('.quick-action-card');

        let messageCount = 0;
        let isProcessing = false;
        let progressInterval = null;

        // ============= UTILITY FUNCTIONS =============

        function updateStats() {
            statsDisplay.textContent = `${messageCount} ${messageCount === 1 ? 'query' : 'queries'}`;
        }

        function hideEmptyState() {
            if (emptyState && emptyState.parentNode) {
                emptyState.style.display = 'none';
            }
        }

        function toggleInput(disabled) {
            sendBtn.disabled = disabled;
            queryInput.disabled = disabled;
            isProcessing = disabled;
        }

        function getTimestamp() {
            return new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        }

        // ============= SMART PROGRESS MESSAGES =============

        function getSmartProgressMessages(userQuery) {
            const query = userQuery.toLowerCase();
            
            // Detect SQL-type queries (database retrieval)
            if (query.includes('link') || query.includes('syllabus') || 
                query.includes('show') || query.includes('list') || 
                query.includes('give me') || query.includes('get')) {
                return [
                    "💾 Connecting to database...",
                    "🔍 Searching syllabus & materials...",
                    "📚 Retrieving course data...",
                    "✨ Formatting results..."
                ];
            }
            
            // Detect PINECONE-type queries (analysis/patterns)
            else if (query.includes('how many') || query.includes('times') || 
                     query.includes('important') || query.includes('frequency') ||
                     query.includes('appear') || query.includes('analysis')) {
                return [
                    "🔍 Searching exam papers...",
                    "📊 Analyzing PYQ patterns...",
                    "🧠 Processing with AI...",
                    "✨ Generating insights..."
                ];
            }
            
            // Generic fallback (combined activities)
            else {
                return [
                    "⚡ Analyzing your query...",
                    "🔍 Searching database & papers...",
                    "🧠 Processing with AI...",
                    "✨ Generating response..."
                ];
            }
        }

        function startSmartProgress(userQuery) {
            const messages = getSmartProgressMessages(userQuery);
            let index = 0;
            
            progressIndicator.classList.add('show');
            progressText.textContent = messages[0];
            
            progressInterval = setInterval(() => {
                index = (index + 1) % messages.length;
                progressText.textContent = messages[index];
            }, 2000); // Change message every 2 seconds
        }

        function stopSmartProgress() {
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
            progressIndicator.classList.remove('show');
        }

        // ============= MESSAGE RENDERING =============

        function addMessage(text, isUser = false) {
            hideEmptyState();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.innerHTML = `
                <div class="avatar">${isUser ? '👤' : '🤖'}</div>
                <span style="font-weight: 700;">${isUser ? 'You' : 'AI Assistant'}</span>
                <span style="margin-left: auto; opacity: 0.6; font-size: 0.85rem;">${getTimestamp()}</span>
            `;
            
            const content = document.createElement('div');
            content.textContent = text;
            content.style.lineHeight = '1.7';
            
            bubble.appendChild(header);
            bubble.appendChild(content);
            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            if (isUser) {
                messageCount++;
                updateStats();
            }
            return messageDiv;
        }

        async function typeText(el, text, speed = 15) {
            return new Promise(resolve => {
                // If text is too long (500+ chars), show instantly
                if (text.length > 500) {
                    el.textContent = text;
                    resolve();
                    return;
                }
                
                let i = 0;
                el.textContent = '';
                const timer = setInterval(() => {
                    if (i < text.length) {
                        el.textContent += text.charAt(i);
                        i++;
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    } else {
                        clearInterval(timer);
                        resolve();
                    }
                }, speed);
            });
        }

        async function addAssistantMessage(data) {
            hideEmptyState();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            
            const bubble = document.createElement('div');
            bubble.className = 'message-bubble';
            
            const header = document.createElement('div');
            header.className = 'message-header';
            header.innerHTML = `
                <div class="avatar">🤖</div>
                <span style="font-weight: 700;">AI Assistant</span>
                <span style="margin-left: auto; opacity: 0.6; font-size: 0.85rem;">${getTimestamp()}</span>
            `;
            
            bubble.appendChild(header);
            
            // Add Badge (Type Indicator)
            if (data.type) {
                const badge = document.createElement('div');
                badge.className = `badge ${data.type.toLowerCase()}`;
                const icons = {
                    'SQL': '💾',
                    'PINECONE': '🔍',
                    'INVALID': '⚠️'
                };
                const icon = icons[data.type] || '🧠';
                badge.innerHTML = `${icon} ${data.type}`;
                bubble.appendChild(badge);
            }
            
            // Add Info Tags (for PINECONE)
            if (data.matches_found !== undefined || data.subject_filter) {
                const tags = document.createElement('div');
                tags.className = 'info-tags';
                
                if (data.subject_filter) {
                    const tag = document.createElement('span');
                    tag.className = 'info-tag';
                    tag.textContent = `📚 ${data.subject_filter}`;
                    tags.appendChild(tag);
                }
                
                if (data.matches_found !== undefined) {
                    const tag = document.createElement('span');
                    tag.className = 'info-tag';
                    tag.textContent = `✨ ${data.matches_found} matches`;
                    tags.appendChild(tag);
                }
                
                bubble.appendChild(tags);
            }
            
            const answer = document.createElement('div');
            answer.style.marginTop = '1rem';
            answer.style.lineHeight = '1.7';
            bubble.appendChild(answer);
            
            messageDiv.appendChild(bubble);
            chatContainer.appendChild(messageDiv);
            
            // Type out the answer with animation
            // await typeText(answer, data.answer, 15);
            // Render markdown directly instead of plain typing
            answer.innerHTML = marked.parse(data.answer || '');

            
            // Add Raw Data Toggle (if data exists)
            const rawData = data.raw_data || data.top_matches || (data.error ? {error: data.error, suggestion: data.suggestion} : null);
            
            if (rawData) {
                const toggle = document.createElement('div');
                toggle.className = 'raw-data-toggle';
                toggle.innerHTML = '📊 View Raw Data';
                
                const rawContent = document.createElement('div');
                rawContent.className = 'raw-data-content';
                rawContent.textContent = JSON.stringify(rawData, null, 2);
                
                toggle.addEventListener('click', () => {
                    rawContent.classList.toggle('show');
                    toggle.innerHTML = rawContent.classList.contains('show') ? '📊 Hide Raw Data' : '📊 View Raw Data';
                });
                
                bubble.appendChild(toggle);
                bubble.appendChild(rawContent);
            }
            
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // ============= API INTEGRATION =============

        async function callBackendAPI(userQuery) {
            try {
                const response = await fetch(API_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        query: userQuery,
                        session_id: session_id
                    })
                });

                // Handle HTTP errors
                if (!response.ok) {
                    if (response.status === 400) {
                        // Bad request (INVALID query)
                        const errorData = await response.json();
                        return {
                            success: false,
                            type: errorData.type || 'INVALID',
                            answer: errorData.error || 'Please ask your query with proper details!',
                            error: errorData.error,
                            suggestion: errorData.suggestion
                        };
                    } else if (response.status === 500) {
                        throw new Error('Server error. Please try again.');
                    } else {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                }

                const data = await response.json();
                
                // Handle null or empty answer
                if (!data.answer || data.answer.trim() === '') {
                    data.answer = 'Please ask your query with proper details!';
                }
                
                return data;

            } catch (error) {
                console.error('API Error:', error);
                
                // Network/connection errors
                if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                    return {
                        success: false,
                        type: 'ERROR',
                        answer: '❌ Cannot connect to server. Please check if backend is running on http://localhost:5000',
                        error: error.message
                    };
                }
                
                // Other errors
                return {
                    success: false,
                    type: 'ERROR',
                    answer: `⚠️ ${error.message}`,
                    error: error.message
                };
            }
        }

        // ============= MAIN HANDLER =============

        async function handleSend() {
            const query = queryInput.value.trim();
            if (!query || isProcessing) return;

            try {
                // 1. Add user message
                addMessage(query, true);
                queryInput.value = '';

                // 2. Disable input & start smart progress
                toggleInput(true);
                startSmartProgress(query);

                // 3. Call real backend API
                const responseData = await callBackendAPI(query);

                // 4. Stop progress indicator
                stopSmartProgress();

                // 5. Add assistant response
                await addAssistantMessage(responseData);

            } catch (error) {
                console.error('Unexpected error:', error);
                stopSmartProgress();
                await addAssistantMessage({
                    type: 'ERROR',
                    answer: '⚠️ An unexpected error occurred. Please try again.',
                    error: error.message
                });
            } finally {
                // 6. Re-enable input
                toggleInput(false);
            }
        }

        // ============= EVENT LISTENERS =============

        // Send button click
        sendBtn.addEventListener('click', handleSend);

        // Enter key press
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
        });

        // Quick action cards
        quickActionCards.forEach(card => {
            card.addEventListener('click', () => {
                const query = card.getAttribute('data-query');
                if (query && !isProcessing) {
                    queryInput.value = query;
                    handleSend();
                }
            });
        });

        // Clear button
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear all messages?')) {
                chatContainer.innerHTML = '';
                emptyState.style.display = 'block';
                chatContainer.appendChild(emptyState);
                messageCount = 0;
                updateStats();
            }
        });

        // ============= INITIALIZATION =============
        updateStats();
