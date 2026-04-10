This repository contains the implementation of a geometry-agnostic sound source localization (SSL) model for planar microphone arrays. 
The model operates on a graph representation of the microphone array and is designed to generalize across unseen array geometries and varying numbers of microphones without retraining.

The architecture consists of three sequential modules: a Message Passing Neural Network (MPNN) that encodes spatial and acoustic relationships between microphones into per-microphone embeddings, a Transformer encoder that contextualizes these embeddings across the full array, and an MLP that regresses the location and strength of the dominant source. 

The model is permutation-invariant with respect to microphone ordering and variable in the number of input channels by construction.
