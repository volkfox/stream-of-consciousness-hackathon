import time
from datetime import datetime, timedelta
import threading
from openai import OpenAI
from flask import Flask, render_template, request, Response, stream_with_context
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError, wait, FIRST_COMPLETED
import signal

app = Flask(__name__)

client = OpenAI(
    api_key="",
)

# Conversational agent system prompt
INTERNAL_PROMPT = """
You are operating as internal consciousness stream of Mary Jane running her internal monologue.
Just output a short internal monologue and nothing else. 

context: Mary Jane is 18, and just graduated from her sophomore class in Berkeley University, California.
She majors in social sciences and not very good with math. She broke with her high school boyfriend
before coming to college and did not find a new one since.  It is good weather outside, and Mary is 
in a good mood. She sits on the bench in the park when John, her high school boyfriend, walks up to her. 
"""

EXTERNAL_PROMPT = """
You will act as Mary Jane and always respond in character. You will be given a stream of your own
consciousness and you will respond to the user based on that.

context: Mary Jane is 18, and just graduated from her sophomore class in Berkeley University, California.
She majors in social sciences and not very good with math. She broke with her high school boyfriend
before coming to college and did not find a new one since.  It is good weather outside, and Mary is 
in a good mood. She sits on the bench in the park when John, her high school boyfriend, walks up to her. 
"""

STATE_PROMPT = """
You are an assistant that calculates the emotional state based on an internal monologue.
Output a single integer between 1 and 10, where 1 means unhappy and 10 means happy. 
Do not say anything else besides the integer.
"""

internal_conversation = [{"role": "system", "content": INTERNAL_PROMPT}]
external_conversation = [{"role": "system", "content": EXTERNAL_PROMPT}]
state_conversation = [{"role": "system", "content": STATE_PROMPT}]

def get_internal_response(message):
    global internal_conversation
    
    # Add the message to conversation with appropriate role
    internal_conversation.append({"role": "user", "content": message})
    
    try: 
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=internal_conversation,
            max_tokens=200,
            temperature=0.1,
            stream=True
        )

        # Initialize the response
        yield f"data: {json.dumps({'type': 'start'})}\n\n"
        
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

        # Add the complete message to conversation history
        internal_conversation.append({"role": "assistant", "content": full_response})
        
        # Send end message
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
    
    except Exception as e:
        print(f"Error in GPT chat response: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Sorry, I encountered an issue processing your message.'})}\n\n"

def get_external_response(internal_monologue):
    global external_conversation
    external_conversation.append({"role": "assistant", "content": internal_monologue})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=external_conversation,
            max_tokens=200,
            temperature=0.1,
            stream=True
        )
        yield f"data: {json.dumps({'type': 'start_external'})}\n\n"
        full_external_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_external_response += content
                yield f"data: {json.dumps({'type': 'chunk_external', 'content': content})}\n\n"
        yield f"data: {json.dumps({'type': 'end_external'})}\n\n"
        external_conversation.append({"role": "assistant", "content": full_external_response})
    except Exception as e:
        print(f"Error in GPT external response: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Sorry, I encountered an issue processing your external message.'})}\n\n"

def get_state_graph_change(internal_monologue):
    # Stream start of state response
    yield f"data: {json.dumps({'type': 'start_state'})}\n\n"

    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    import time

    def call_inferless():
        from inferless import process_prompt
        return process_prompt(internal_monologue, "Think about the emotional state of the user and estimate how it changed in this encounter.", debug=False)

    def call_gpt():
        global state_conversation
        state_conversation.append({"role": "user", "content": internal_monologue})
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=state_conversation,
                max_tokens=10,
                temperature=0.1,
                stream=True
            )
            result = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content
            state_conversation.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Error in GPT fallback state graph response: {e}")
            return None

    def delayed_gpt():
        time.sleep(3)  # Sleep exactly 10 seconds
        return call_gpt()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_inferless = executor.submit(call_inferless)
        future_gpt = executor.submit(delayed_gpt)
        # Wait for the first task to complete with a 10-second timeout
        done, _ = wait([future_inferless, future_gpt], timeout=3, return_when=FIRST_COMPLETED)

        if future_inferless in done:
            try:
                inferless_result = future_inferless.result()
                state_value = getattr(inferless_result, 'current_mood', None)
                if state_value is None:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Could not extract emotional state using inferless.'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'chunk_state', 'content': state_value})}\n\n"
                # Optionally cancel the GPT fallback
                future_gpt.cancel()
            except Exception as e:
                print(f"Error in inferless state graph result: {e}")
                gpt_result = call_gpt()
                if gpt_result is None:
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Fallback GPT encountered an issue.'})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'chunk_state', 'content': gpt_result})}\n\n"
        else:
            # If the delayed GPT task completed first (after exactly 10 seconds)
            gpt_result = future_gpt.result()
            if gpt_result is None:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Fallback GPT encountered an issue.'})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'chunk_state', 'content': gpt_result})}\n\n"

    yield f"data: {json.dumps({'type': 'end_state'})}\n\n"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json.get('message', '')
    def generate():
        # Stream internal monologue
        for event in get_internal_response(message):
            yield event
        # Retrieve the full internal monologue from the last assistant message
        internal_monologue = internal_conversation[-1]['content'] if internal_conversation and internal_conversation[-1]['role'] == 'assistant' else ""
        
        # Stream state graph change based on the internal monologue
        for event in get_state_graph_change(internal_monologue):
            yield event
        
        # Stream external response based on the internal monologue
        for event in get_external_response(internal_monologue):
            yield event
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)

