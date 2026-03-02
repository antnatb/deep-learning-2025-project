# Deep Learning Project 2025 - Few-Shot Adaptation of CLIP

**Università di Trento**

## Overview

This project investigates few-shot adaptation methods for CLIP (Contrastive Language-Image Pre-training) on fine-grained image classification tasks, using the Oxford Flowers 102 dataset.

## Key Findings

We explored several parameter-efficient fine-tuning (PEFT) approaches including CoOp, CoCoOp, and KgCoOp. Our investigation revealed that **label engineering**—aligning class names with CLIP's training vocabulary—can be as impactful as sophisticated adaptation methods, significantly improving zero-shot accuracy.

## Contents

- `report.ipynb` — Complete project notebook with code, experiments, results, and analysis

## Authors

- Antonio N. Bruno (ID: 258035)
- Edoardo Di Tommaso (ID: 258433)

## References

1. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
2. Nilsback & Zisserman, "Automated Flower Classification over a Large Number of Classes"
3. Zhou et al., "Learning to Prompt for Vision-Language Models" (CoOp)
4. Zhou et al., "Conditional Prompt Learning for Vision-Language Models" (CoCoOp)
5. Yao et al., "Visual-Language Prompt Tuning with Knowledge-guided Context Optimization" (KgCoOp)
