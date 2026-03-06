# Day 1 Scope Lock

## One feature
**Earnings Call Signal Extraction Engine**

## What it is
A tool that converts an earnings call into a structured signal summary for a retail trader.

## Primary user
Retail traders who follow earnings announcements and want a faster, more structured way to detect meaningful changes in management tone and guidance.

## Core pain
Retail traders cannot reliably process long, noisy earnings calls fast enough to spot the few signal-bearing moments that matter, especially changes in guidance, confidence, and tone.

## Product promise
Turn a long earnings call into a fast, structured, auditable signal summary that helps a retail trader see what changed, why it matters, and where to look first.

## One-sentence problem
Retail traders struggle to extract the few important signal-bearing moments from long, noisy earnings calls quickly enough to act with confidence.

## One-sentence value proposition
The Earnings Call Signal Extraction Engine turns a complex earnings call into a fast, structured, auditable signal summary that highlights guidance shifts, tone changes, and the evidence behind them.

## Baseline definition
A deterministic pipeline that processes one earnings call and outputs sentiment trend, extracted guidance statements, and a simple structured summary.

## Improved version definition
An enhanced version of the pipeline that compares current vs prior guidance, detects tone shifts, and surfaces clearer evidence-backed signal changes for trader review.

## What the user sees in the demo
The user selects a sample earnings call, clicks analyze, and receives:
- an overall signal summary
- extracted guidance statements
- guidance change vs prior call
- tone-change moments
- evidence snippets from the transcript
- a short “what to review first” summary

## Explicitly out of scope
- autonomous trading bot
- portfolio recommendations
- full forecasting engine
- hedge-fund-grade analytics platform
- production deployment
- broad multimodal research suite beyond the capstone demo

## Rough slide titles
1. The Problem: Earnings Calls Are High-Value but Hard to Process
2. Target User: Retail Traders Need Faster Signal Detection
3. Our Solution: Earnings Call Signal Extraction Engine
4. Reverse Scope: From User Need to Data Feature
5. Baseline Feature: Sentiment + Guidance Extraction
6. Improved Feature: Guidance Revision + Tone Change Detection
7. Feasibility: Data, Model, and Risk
8. Demo: From Call to Structured Signal Summary
9. Roadmap: Now, Next, Later
10. Learnings, Limitations, and Next Steps
