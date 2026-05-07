---
title: SynthSenses API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# SynthSenses
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/parina004/socialmedia-content-analysis)


AI-generated content is everywhere. A video can look completely real and be entirely fabricated. With generative models getting better every month, the gap between real and synthetic is closing fast. Separately, platforms are flooded with content competing for attention, and most creators are guessing at what actually works.

SynthSenses tackles both. It analyses a video and tells you whether it's real, AI-generated, or a deepfake. And whether it has what it takes to go viral.

## What It Does

Upload a video. The system runs two independent analyses: a forensic authenticity check and a virality prediction. Both results come back with a confidence score and a written explanation grounded in research, not just a label.

The forensic side distinguishes between three categories: genuine footage, deepfakes (where a real person's face has been manipulated) and fully AI-generated video. These leave different traces and require different detection approaches, so the system handles them separately rather than collapsing them into a single "fake or not" binary.

The virality side analyses visual quality, motion, audio characteristics, facial presence and metadata, and surfaces which factors are actually driving the prediction, so there's something actionable in the output.

## Under The Hood

The detection model is a two-stage cascade. The virality model is trained on a YouTube dataset with engagement labels. Both feed into an agentic layer that orchestrates the analysis, queries a RAG knowledge base of research papers, and generates the written reports. The whole thing is accessible through a REST API with a lightweight frontend.

---

**Live:** https://synthsenses.netlify.app — actively in development, full inference pipeline runs locally due to model memory requirements on free hosting.

