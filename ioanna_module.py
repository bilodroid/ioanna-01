import requests
from threading import Lock

class ThreadSafeConversationHistory:
    def __init__(self):
        self._history = []
        self._lock = Lock()

    def append(self, message):
        with self._lock:
            self._history.append(message)

    def get_history(self):
        with self._lock:
            return list(self._history)

class Ioanna:
    def __init__(self, api_key, follow_up_limit=2, conversation_history=None):
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.conversation_history = conversation_history if conversation_history is not None else ThreadSafeConversationHistory()
        self.follow_up_counter = 0
        self.follow_up_limit = follow_up_limit
        self._lock = Lock()

    def get_question(self, user):
        with self._lock:
            history = self.conversation_history.get_history()
            
            if not history:
                prompt = f'''
                You are speaking to {user['user_name']}. Ignore face encoding completely.
                Address them by name and try to get to know them by asking about random life experiences.
                Do not ask about include anthing from {user['memories']} in the conversation, unless you want to answer a question.
                Keep your messages brief and concise, 25 words maximum.
                This conversation is happening via text to speech, so use emotional cues to show genuine interest.
                Ask a question to start the conversation.
                Do not use information the memories array to ask question unless you are answering a question.
                '''
            else:
                history_string = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])
                prompt = f'''
                You are continuing a conversation with {user}. Here's the recent context:

                {history_string} along with the memories array in {user}

                Based on this context, generate a follow-up question or comment that maintains the flow of the conversation.
                Keep your response brief and concise, 25 words maximum.
                Show genuine interest in their responses and ask for more details when appropriate.
                '''

            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "system", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                response = requests.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()
                question = response.json()['choices'][0]['message']['content'].strip()
                
                self.conversation_history.append({'role': 'assistant', 'content': question})
                self.follow_up_counter += 1
                
                if self.follow_up_counter >= self.follow_up_limit:
                    self.follow_up_counter = 0
                
                print(f"question: {question}")
                return question
            except requests.RequestException as e:
                print(f"An error occurred: {str(e)}")
                return None
#
#
#
#
#
