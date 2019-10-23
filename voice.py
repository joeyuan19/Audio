import speech_recognition as s_r

names = s_r.Microphone.list_microphone_names()
print(names)

mic_idx = 0

mic = s_r.Microphone(device_index=mic_idx)
r   = s_r.Recognizer()

with mic as source:
    print("say hello")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
s = r.recognize_google(audio)

print(s)

