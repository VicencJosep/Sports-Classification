# SOTA Detección Highlights en Fútbol (2025-2026)

## 1. VLMs y Temporal Grounding (Semántica)
- **TIMEREFINE (WACV, 2026)**: Video LLM que refina iterativamente las fronteras temporales de una jugada sin timestamps fijos. [Búsqueda arXiv](https://arxiv.org/search/cs?query=TIMEREFINE)
- **UniTime (NeurIPS 2025)**: VLM (Qwen2) que usa tokens de tiempo para devolver inicio/fin de jugadas a partir de prompts textuales. [Búsqueda arXiv](https://arxiv.org/search/cs?query=UniTime+Universal+Temporal+Grounding)
- **SVHighlights (Junio 2026)**: Segmenta el video guiándose por el audio del comentarista y usa *Dense Captioning* para evaluar la jugada. [Búsqueda arXiv](https://arxiv.org/search/cs?query=SVHighlights)
- **TEMPURA (Sept. 2026)**: Segmentación vía *Dense Captioning*; aprende progresiones lógicas largas (ej. pase->regate->tiro). [Búsqueda arXiv](https://arxiv.org/search/cs?query=TEMPURA+Dense+Captioning)
- **SoccerChat (Mayo 2025)**: Framework multimodal para evaluar clips de fútbol densos (fine-tuning contra oclusiones). [arXiv:2505.00000](https://arxiv.org/)
- **Soccer-CLIP / SoccerLens (2026)**: Integra visión/texto y fuerza al modelo a justificar decisiones basándose en cinemática (jugadores), no en gráficos/cortes. [arXiv:2605.09598](https://arxiv.org/abs/2605.09598)

## 2. Grafos y Tracking (Geometría)
- **TraVIS (2025)**: Fusión de grafos (GNN) de tracking con redes de video para aprender la "tensión geométrica" implícitamente sin reglas estáticas. [Búsqueda arXiv](https://arxiv.org/search/cs?query=TraVIS)
- **Ball-Free Event Detection (arXiv, 2026)**: Inferencia de eventos usando topología sobre rutas probables de posesión de jugadores (no depende del balón). [Búsqueda arXiv](https://arxiv.org/search/cs?query=Ball-Free+Soccer+Event+Detection)
- **Player-Centric Action Spotting (SoccerNet, 2026)**: Aplica crop dinámico sobre el jugador con balón para eliminar el ruido visual del plano general. [arXiv:2606.28389](https://arxiv.org/abs/2606.28389)
- **Review CV Tech Football (MDPI, 2025)**: Revisión general sobre tracking continuo de jugadores y oclusión. [MDPI](https://www.mdpi.com/search?q=Computer+Vision+Technology+for+Football+Videos)

## 3. Clásicos y CNNs (Clasificación Espaciotemporal)
- **Auto Soccer Highlight Detection (Stanford)**: CNNs 3D (R3D, ViT) en clips de 5s. Clasificación binaria pura. [automatic-soccer-game-highlight-detection.pdf]()
- **Soccer Video Summarization (MIPR, 2019)**: Arquitectura 3D-ResNet34 + LSTM para cadenas temporales. [IEEE](https://ieeexplore.ieee.org/document/8919375)
- **Detecting key Soccer match events (arXiv, 2022)**: Faster R-CNN/YOLOv5 frame a frame. Bueno en balón parado, malo en transiciones. [Búsqueda arXiv](https://arxiv.org/search/cs?query=Detecting+key+Soccer+match+events+highlights)

## 4. Multimodalidad y Audio
- **Highlights physiological responses (2023) / Multimodal (2026)**: Cruza audio (emoción) y biometría para encontrar hitos que la imagen no capta (ej. tiros al palo en 0-0). [Google Scholar](https://scholar.google.com/scholar?q=Highlights+detection+in+movie+scenes+through+inter-users+physiological+responses)