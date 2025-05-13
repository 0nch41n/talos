# TextGeneratorLLM: On-Chain Neural Language Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TextGeneratorLLM is a sophisticated on-chain mini language model implemented entirely in Solidity. It brings modern neural language generation techniques to blockchain environments by simulating key components of transformer architecture within the constraints of the EVM.

## 🧠 Architecture Overview

This contract implements a novel approach to on-chain language generation through a multi-layered architecture inspired by transformer models:

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  ┌─────────────┐    ┌─────────────┐             │
│  │  Vocabulary │    │ Positional  │             │
│  │  Management │    │  Encoding   │             │
│  └─────────────┘    └─────────────┘             │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌──────────────────────────────────────────┐  │
│  │                                          │  │
│  │      Multi-Layer Transition Networks     │  │
│  │                                          │  │
│  └──────────────────────────────────────────┘  │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌─────────────┐    ┌─────────────┐             │
│  │   Context   │    │  Attention  │             │
│  │  Awareness  │    │ Mechanisms  │             │
│  └─────────────┘    └─────────────┘             │
│          │                 │                    │
│          ▼                 ▼                    │
│  ┌──────────────────────────────────────────┐  │
│  │                                          │  │
│  │     Domain & Sentiment-Aware Output      │  │
│  │                                          │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

## ✨ Key Features

- **Transformer-Inspired Architecture**: Implements up to 6 neural processing layers to simulate transformer-like text generation
- **Attention Mechanism**: Utilizes custom attention weights to model token-to-token relationships and enhance contextual relevance
- **Positional Encoding**: Incorporates position-aware token representations to capture sequence information
- **N-gram Transitions**: Supports both unigram and bigram probabilistic transitions for more coherent text generation
- **Domain & Sentiment Control**: Generates text with specified domain focus and sentiment scores
- **Context-Aware Generation**: Maintains context window of previously generated tokens to influence future token selection
- **Template-Based Generation**: Supports structured output through parameterized templates with intelligent slot filling
- **Role-Based Access Control**: Implements sophisticated permission system with distinct roles for administration, training, and generation

## 🔬 Technical Implementation

### Neural Language Generation Components

The contract simulates neural network operations through several sophisticated mechanisms:

1. **Multi-Layer Processing**
   ```solidity
   function getNextWordWithLayers(
       uint256 seed,
       uint256 prevWordId,
       uint256 wordId,
       uint256[] memory context,
       uint256 contextLength,
       uint8 maxLayers
   ) internal view returns (uint256)
   ```
   This function simulates the forward pass of a multi-layer neural network, with each layer contributing to token selection based on learned transitions and contextual relevance.

2. **Attention-Based Scoring**
   ```solidity
   function calculateAttentionScore(
       uint256 candidateId,
       uint256[] memory context,
       uint256 contextLength,
       uint8 layer
   ) internal view returns (uint256)
   ```
   Implements a simplified attention mechanism to compute relevance scores between tokens and their context, with layer-specific decay factors.

3. **Context-Aware Generation**
   The contract maintains a context window of previously generated tokens, allowing it to make generation decisions that account for sequence history:
   ```solidity
   context[length] = wordId;
   length++;
   ```

### Data Structures

The contract employs several advanced data structures:

1. **Word Metadata**
   ```solidity
   struct WordMetadata {
       uint8 domain;
       uint8 sentiment;
       uint8 partOfSpeech;
       uint8 commonality;
   }
   ```

2. **Layered Transitions**
   ```solidity
   struct LayeredTransition {
       uint256 nextWordId;
       uint256 probability;
       uint8 layer;
   }
   ```

3. **Attention Weights**
   ```solidity
   struct AttentionWeight {
       uint256 contextWordId;
       uint256 weight;
   }
   ```

4. **Positional Encoding**
   ```solidity
   struct PositionMetadata {
       uint256 position;
       uint256 weight;
   }
   ```

## 📊 Performance & Constraints

The model operates within the following constraints:

- Vocabulary size: up to 2,048 tokens
- Max sentence length: 64 tokens
- Max layers: 6 neural processing layers
- Max transitions per token: 30 possible next tokens

Despite these constraints, the model achieves impressive coherence by leveraging sophisticated probabilistic selection algorithms and contextual awareness.

## 🚀 Usage Examples

### Text Generation

```solidity
// Generate text with domain and sentiment control
string memory generatedText = textGenerator.generateTextWithLayers(
    uint256(blockhash(block.number - 1)), // Seed from block hash
    TextGeneratorLLM.DOMAIN_RPG,          // Domain: RPG genre
    2,                                    // Positive sentiment
    false,                                // Free-form generation (not template-based)
    32,                                   // Max length: 32 tokens
    4                                     // Use 4 neural layers
);
```

### Adding Word Transitions

```solidity
// Add a transition from "knight" to "sword" with high probability
uint256 knightId = textGenerator.wordToId("knight") - 1;
uint256[] memory nextWords = new uint256[](1);
nextWords[0] = textGenerator.wordToId("sword") - 1;
uint256[] memory probs = new uint256[](1);
probs[0] = 800; // 80% probability (out of 1000)

textGenerator.setLayeredTransitions(
    2,           // Layer 2
    knightId,    // From "knight"
    nextWords,   // To ["sword"]
    probs        // With probabilities [800]
);
```

## 🔧 Technical Requirements

- Solidity ^0.8.19
- OpenZeppelin Contracts (AccessControl, Pausable)
- Custom blockspace or L2 environment recommended due to computational intensity

## 📈 Future Development

- Implement efficient on-chain training mechanisms
- Expand vocabulary capacity through hierarchical storage
- Add support for more complex attention patterns
- Integrate with other on-chain AI systems

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔍 Academic Context

This implementation draws inspiration from transformer architecture concepts introduced in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), adapted for the unique constraints of blockchain environments.

---

*Note: This is an experimental research project demonstrating the intersection of neural language generation and blockchain technology. While it implements conceptual elements of transformer models, it operates at a significantly reduced scale compared to traditional LLMs.*
