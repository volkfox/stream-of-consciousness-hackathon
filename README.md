# AGI hackathon 2/15

This project explores emotional changes in simulated personas.

Unlike the "traditional" chat, where the personas in LLM are maintained via simple prompting,
this project explores the more complicated setup of three plances represeting a virtual person:

(a) Emotional plane. Changes with environment and conversation flow.
(b) Internal monologue - the stream of thoughts running internally ("deep thinking")
(c) External response - the actual utterances by the simulated persona.


Tech stack >
1. DeepSeekR1-distilled finetuned for roleplaying on Nebius GPU cloud
2. Inferless service as DeepSeek API endpoint
3. OpenAI API for integration
