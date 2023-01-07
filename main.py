import whisper
import openai
import config

model = whisper.load_model("base")
result = model.transcribe("motivation.mp3")

openai.api_key = config.api_key

response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Summarize the following text: " + result["text"],
    temperature=0,
    max_tokens=64,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

print(response)
