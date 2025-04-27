# muffled_audio_detection

Detects muffled audio vs clear audio. 

Run audiogen.py to create muffled audio tts and clear audio tts. Run audio_classifiction.py and simultaneously play the clear or muffled audio to see the classification based on 3 second chunks. Optionally choosing a file and see realtime classification. Find training data at https://commonvoice.mozilla.org/en, save clear audio clips into clear_audio/clips dir. Run audio_training.py to generate muffled_audio/clips. Run audio_training.py to get a new classifier.

