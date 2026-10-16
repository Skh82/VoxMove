# VoxMove

### Joint Speech and Gesture Generation with VoXtream2

**VoxMove** explores synchronized speech and 3D gesture generation by connecting a speech model to a body-aware motion generator. Rather than relying only on audio or text, we extract **intermediate hidden states from VoXtream2** and use them as time-aligned conditioning vectors for gesture generation.

## Demos

https://github.com/user-attachments/assets/5fdd48e8-9a33-4c9d-9e23-57cc1e769ebd

https://github.com/user-attachments/assets/80b3b04a-f86e-4005-8b6f-95094c99e7d5

https://github.com/user-attachments/assets/a39bf128-1a8b-4cdb-b37f-828fe9973ad5

## How it works

- **VoXtream2 conditioning:** Extract internal speech-model states and align them with motion frames. These representations condition the gesture generator on speech-related information already available inside VoXtream2.
- **Body-aware DiTs:** Use specialized Diffusion Transformer branches for different motion components, including body, hands, facial expression, and global translation, with information shared across branches.
- **Temporal generation:** Generate motion in successive windows, using previous motion as context to encourage continuity across the sequence.

## Data

We use **BEAT2** for speech-gesture motion training and also incorporate the **additional set** to improve robustness across speakers, motion styles, and recording variation.

## Project status

VoxMove is an ongoing research project. The videos above show qualitative generation examples; they are not a comprehensive quantitative evaluation.
