# VoxMove

### Joint Speech and Gesture Generation with VoXtream2

**VoxMove** explores synchronized speech and 3D gesture generation by connecting a speech model to a body-aware motion generator. Rather than relying only on audio or text, we extract **intermediate hidden states from VoXtream2** and use them as time-aligned conditioning vectors for gesture generation.

## Demos

https://github.com/user-attachments/assets/5fdd48e8-9a33-4c9d-9e23-57cc1e769ebd

https://github.com/user-attachments/assets/80b3b04a-f86e-4005-8b6f-95094c99e7d5

https://github.com/user-attachments/assets/a39bf128-1a8b-4cdb-b37f-828fe9973ad5
## Speaking-Rate Experiment

### One text, three speaking rates

How does changing VoXtream2’s speaking rate affect the generated speech and accompanying motion?

This experiment connects a frozen VoXtream2 speech model to the trained VoxMove motion generator. Speech is generated at different target rates, and its internal hidden states condition the generation of gestures and facial movements.

**Voice reference:** Scott   
**Target speaking rates:** 1, 4, and 7 syllables per second

### Input text

> Hi, I'm VoxMove. I turn written text into speech and full-body animation.
> VoXtream two generates my voice, while a motion model uses its internal
> speech features to create gestures and facial movements.
> I generate movement in short chunks, using the previous motion to guide
> each new chunk. For this demonstration, my voice is based on Scott.

### Rate 1 · Slow


https://github.com/user-attachments/assets/66e85f79-8125-40a9-805a-7193dcb88eaf

### Rate 4 · Intermediate


https://github.com/user-attachments/assets/26867a6a-21b2-4c7f-8ca3-f0f0240b96cd


### Rate 7 · Fast


https://github.com/user-attachments/assets/11bb6b81-26d6-4ee1-98e7-4e9eb77bba58


These are qualitative examples of coupled speech and motion generation.
Speaking-rate values are targets in syllables per second, not playback-speed
multipliers. The examples do not establish that motion quality improves or
degrades consistently with speaking rate.

## How it works

- **VoXtream2 conditioning:** Extract internal speech-model states and align them with motion frames. These representations condition the gesture generator on speech-related information already available inside VoXtream2.
- **Body-aware DiTs:** Use specialized Diffusion Transformer branches for different motion components, including body, hands, facial expression, and global translation, with information shared across branches.
- **Temporal generation:** Generate motion in successive windows, using previous motion as context to encourage continuity across the sequence.

## Data

We use **BEAT2** for speech-gesture motion training and also incorporate the **additional set** to improve robustness across speakers, motion styles, and recording variation.

## Project status

VoxMove is an ongoing research project. The videos above show qualitative generation examples; they are not a comprehensive quantitative evaluation.
