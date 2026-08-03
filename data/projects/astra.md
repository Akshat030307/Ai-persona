---
title: ASTRA
slug: astra
order: 4
tags: [Voice AI, Offline AI, LLM]
stack: [LLaMA-3.2, Vosk, SQLite, PyTorch]
github_url: https://github.com/Akshat030307/ASTRA
drive_video_url: ""
summary: An offline, Jarvis-inspired voice assistant with dual local LLM inference, zero data egress, and real-time screen OCR.
---
ASTRA is a Jarvis-inspired voice assistant that operates fully offline, via process-isolated dual
LLM inference — a 3B resident model paired with an 8B lazy-loaded fallback — with SQLite handling
state persistence.

It integrates offline speech recognition (Vosk), local document search, and an intent
classification engine that drives 35+ automated system controls. It also does real-time screen OCR
analysis and multi-turn conversation tracking, guaranteeing complete local privacy with zero data
egress — nothing leaves the machine.

<!-- TODO: paste a Google Drive share link for the ASTRA demo video into drive_video_url above,
     if/when one exists. Add detail on the process-isolation architecture between the two models,
     and what motivated the fully local-first / zero-egress design. -->
