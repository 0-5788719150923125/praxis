curl -X POST http://localhost:2100/messages \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant trained to complete texts, answer questions, and engage in conversation."
    },
    {
      "role": "developer",
      "content": "Write thy wrong."
    },
    {
      "role": "user",
      "content": "Is there anybody there?"
    }
  ],
  "max_new_tokens": 256,
  "temperature": 0.4,
  "repetition_penalty": 1.15,
  "do_sample": true,
  "use_cache": false,
  "skip_special_tokens": false
}'
