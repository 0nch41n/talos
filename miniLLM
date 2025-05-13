// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/utils/Strings.sol";

contract TextGeneratorLLM is AccessControl, Pausable {
    using Strings for uint256;

    // Roles
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant TRAINER_ROLE = keccak256("TRAINER_ROLE");
    bytes32 public constant GENERATOR_ROLE = keccak256("GENERATOR_ROLE");

    // Constants
    uint256 public constant MAX_VOCAB_SIZE = 2048; // Increased for richer vocabulary
    uint256 public constant MAX_TEMPLATES = 200;
    uint256 public constant MAX_SLOTS_PER_TEMPLATE = 12;
    uint256 public constant MAX_TRANSITIONS_PER_WORD = 30;
    uint256 public constant MAX_SENTENCE_LENGTH = 64;
    uint256 public constant SCALE_FACTOR = 1000;
    uint8 public constant MAX_LAYERS = 6; // Simulate transformer layers

    // Domain types
    uint8 public constant DOMAIN_GENERAL = 0;
    uint8 public constant DOMAIN_RPG = 10;

    // Sentiment classes
    uint8 public constant CLASS_VERY_NEGATIVE = 0;
    uint8 public constant CLASS_NEUTRAL = 3;
    uint8 public constant CLASS_VERY_POSITIVE = 6;

    // Vocabulary storage
    mapping(uint256 => string) public vocabulary;
    mapping(string => uint256) public wordToId;
    uint256 public vocabSize;

    // Word metadata
    struct WordMetadata {
        uint8 domain;
        uint8 sentiment;
        uint8 partOfSpeech; // 0=noun, 1=verb, 2=adjective, 3=adverb, 4=other
        uint8 commonality;
    }
    mapping(uint256 => WordMetadata) public wordMetadata;

    // Positional encoding
    struct PositionMetadata {
        uint256 position; // Position in sequence
        uint256 weight;   // Scaled weight
    }
    mapping(uint256 => PositionMetadata[]) public wordPositionMetadata;

    // Layered transitions
    struct LayeredTransition {
        uint256 nextWordId;
        uint256 probability;
        uint8 layer;
    }
    mapping(uint8 => mapping(uint256 => LayeredTransition[])) public layeredWordTransitions;
    mapping(uint8 => mapping(uint256 => mapping(uint256 => LayeredTransition[]))) public layeredBigramTransitions;

    // Attention weights
    struct AttentionWeight {
        uint256 contextWordId;
        uint256 weight; // Scaled attention weight
    }
    mapping(uint256 => AttentionWeight[]) public attentionWeights;

    // Sentence templates
    struct SentenceTemplate {
        string template;
        uint8 domain;
        uint8 sentiment;
        uint8[MAX_SLOTS_PER_TEMPLATE] slotTypes;
        uint8 slotCount;
        bool active;
    }
    SentenceTemplate[] public templates;
    mapping(uint8 => uint256[]) public domainTemplates;
    mapping(int8 => uint256[]) public sentimentTemplates;

    // Starting words
    mapping(uint8 => uint256[]) public domainStartWords;
    mapping(int8 => uint256[]) public sentimentStartWords;

    // Events
    event TextGenerated(address indexed generator, string text, uint8 domain, int8 sentiment);
    event TemplateAdded(uint256 templateId, string template, uint8 domain, int8 sentiment);
    event VocabularyExpanded(uint256 newWords, address indexed trainer);
    event TransitionsUpdated(uint8 layer, uint256 wordId, uint256 transitionCount);
    event AttentionWeightsUpdated(uint256 wordId, uint256 contextCount);

    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(ADMIN_ROLE, msg.sender);
        _setupRole(TRAINER_ROLE, msg.sender);
        _setupRole(GENERATOR_ROLE, msg.sender);
    }

    // Vocabulary Management
    function addWords(
        string[] calldata words,
        uint8[] calldata domains,
        int8[] calldata sentiments,
        uint8[] calldata partsOfSpeech,
        uint8[] calldata commonalities
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(words.length == domains.length && words.length == sentiments.length, "Array length mismatch");
        require(vocabSize + words.length <= MAX_VOCAB_SIZE, "Exceeds max vocabulary");

        uint256 addedCount = 0;
        for (uint256 i = 0; i < words.length; i++) {
            if (wordToId[words[i]] != 0) continue;
            require(domains[i] <= DOMAIN_RPG, "Invalid domain");
            require(sentiments[i] >= -5 && sentiments[i] <= 5, "Invalid sentiment");
            require(partsOfSpeech[i] <= 4, "Invalid part of speech");
            require(commonalities[i] >= 1 && commonalities[i] <= 10, "Invalid commonality");

            uint256 wordId = vocabSize;
            vocabulary[wordId] = words[i];
            wordToId[words[i]] = wordId + 1;

            wordMetadata[wordId] = WordMetadata({
                domain: domains[i],
                sentiment: uint8(int8(sentiments[i]) + 5),
                partOfSpeech: partsOfSpeech[i],
                commonality: commonalities[i]
            });

            if (commonalities[i] >= 7) domainStartWords[domains[i]].push(wordId);
            sentimentStartWords[sentiments[i]].push(wordId);

            vocabSize++;
            addedCount++;
        }

        emit VocabularyExpanded(addedCount, msg.sender);
    }

    // Positional Metadata
    function setPositionMetadata(
        uint256 wordId,
        uint256[] calldata positions,
        uint256[] calldata weights
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(wordId < vocabSize, "Invalid word ID");
        require(positions.length == weights.length, "Array length mismatch");
        delete wordPositionMetadata[wordId];

        for (uint256 i = 0; i < positions.length; i++) {
            require(positions[i] < MAX_SENTENCE_LENGTH, "Invalid position");
            wordPositionMetadata[wordId].push(PositionMetadata({
                position: positions[i],
                weight: weights[i]
            }));
        }
    }

    // Layered Transitions
    function setLayeredTransitions(
        uint8 layer,
        uint256 wordId,
        uint256[] calldata nextWordIds,
        uint256[] calldata probabilities
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(layer < MAX_LAYERS, "Invalid layer");
        require(wordId < vocabSize, "Invalid word ID");
        require(nextWordIds.length == probabilities.length, "Array length mismatch");
        require(nextWordIds.length <= MAX_TRANSITIONS_PER_WORD, "Too many transitions");

        delete layeredWordTransitions[layer][wordId];
        uint256 totalProb = 0;
        for (uint256 i = 0; i < nextWordIds.length; i++) {
            require(nextWordIds[i] < vocabSize, "Invalid next word ID");
            totalProb += probabilities[i];
            layeredWordTransitions[layer][wordId].push(LayeredTransition({
                nextWordId: nextWordIds[i],
                probability: probabilities[i],
                layer: layer
            }));
        }

        if (totalProb != SCALE_FACTOR && totalProb > 0) {
            for (uint256 i = 0; i < layeredWordTransitions[layer][wordId].length; i++) {
                layeredWordTransitions[layer][wordId][i].probability =
                    (layeredWordTransitions[layer][wordId][i].probability * SCALE_FACTOR) / totalProb;
            }
        }

        emit TransitionsUpdated(layer, wordId, nextWordIds.length);
    }

    function setLayeredBigramTransitions(
        uint8 layer,
        uint256 firstWordId,
        uint256 secondWordId,
        uint256[] calldata nextWordIds,
        uint256[] calldata probabilities
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(layer < MAX_LAYERS, "Invalid layer");
        require(firstWordId < vocabSize && secondWordId < vocabSize, "Invalid word IDs");
        require(nextWordIds.length == probabilities.length, "Array length mismatch");
        require(nextWordIds.length <= MAX_TRANSITIONS_PER_WORD, "Too many transitions");

        delete layeredBigramTransitions[layer][firstWordId][secondWordId];
        uint256 totalProb = 0;
        for (uint256 i = 0; i < nextWordIds.length; i++) {
            require(nextWordIds[i] < vocabSize, "Invalid next word ID");
            totalProb += probabilities[i];
            layeredBigramTransitions[layer][firstWordId][secondWordId].push(LayeredTransition({
                nextWordId: nextWordIds[i],
                probability: probabilities[i],
                layer: layer
            }));
        }

        if (totalProb != SCALE_FACTOR && totalProb > 0) {
            for (uint256 i = 0; i < layeredBigramTransitions[layer][firstWordId][secondWordId].length; i++) {
                layeredBigramTransitions[layer][firstWordId][secondWordId][i].probability =
                    (layeredBigramTransitions[layer][firstWordId][secondWordId][i].probability * SCALE_FACTOR) / totalProb;
            }
        }

        emit TransitionsUpdated(layer, secondWordId, nextWordIds.length);
    }

    // Attention Weights
    function setAttentionWeights(
        uint256 wordId,
        uint256[] calldata contextWordIds,
        uint256[] calldata weights
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(wordId < vocabSize, "Invalid word ID");
        require(contextWordIds.length == weights.length, "Array length mismatch");
        delete attentionWeights[wordId];

        for (uint256 i = 0; i < contextWordIds.length; i++) {
            require(contextWordIds[i] < vocabSize, "Invalid context word ID");
            attentionWeights[wordId].push(AttentionWeight({
                contextWordId: contextWordIds[i],
                weight: weights[i]
            }));
        }

        emit AttentionWeightsUpdated(wordId, contextWordIds.length);
    }

    // Template Management
    function addTemplate(
        string calldata templateStr,
        uint8 domain,
        int8 sentiment,
        uint8[] calldata slotTypes
    ) external onlyRole(TRAINER_ROLE) whenNotPaused {
        require(templates.length < MAX_TEMPLATES, "Max templates reached");
        require(domain <= DOMAIN_RPG, "Invalid domain");
        require(sentiment >= -5 && sentiment <= 5, "Invalid sentiment");
        require(slotTypes.length <= MAX_SLOTS_PER_TEMPLATE, "Too many slots");

        uint8[MAX_SLOTS_PER_TEMPLATE] memory slotTypesFixed;
        for (uint8 i = 0; i < slotTypes.length; i++) {
            require(slotTypes[i] <= 4, "Invalid slot type");
            slotTypesFixed[i] = slotTypes[i];
        }

        SentenceTemplate memory newTemplate = SentenceTemplate({
            template: templateStr,
            domain: domain,
            sentiment: uint8(int8(sentiment) + 5),
            slotTypes: slotTypesFixed,
            slotCount: uint8(slotTypes.length),
            active: true
        });

        templates.push(newTemplate);
        uint256 templateId = templates.length - 1;
        domainTemplates[domain].push(templateId);
        sentimentTemplates[sentiment].push(templateId);

        emit TemplateAdded(templateId, templateStr, domain, sentiment);
    }

    // Text Generation
    function generateTextWithLayers(
        uint256 seed,
        uint8 domain,
        int8 sentiment,
        bool useTemplates,
        uint256 maxLength,
        uint8 maxLayers
    ) external whenNotPaused returns (string memory) {
        require(hasRole(GENERATOR_ROLE, msg.sender), "Must have generator role");
        require(domain <= DOMAIN_RPG, "Invalid domain");
        require(sentiment >= -5 && sentiment <= 5, "Invalid sentiment");
        require(maxLength <= MAX_SENTENCE_LENGTH, "Length exceeds maximum");
        require(maxLayers <= MAX_LAYERS, "Too many layers");

        string memory generatedText;
        if (useTemplates && templates.length > 0) {
            generatedText = generateFromTemplate(seed, domain, sentiment);
        } else {
            generatedText = generateFromTransitions(seed, domain, sentiment, maxLength, maxLayers);
        }

        emit TextGenerated(msg.sender, generatedText, domain, sentiment);
        return generatedText;
    }

    function generateFromTemplate(
        uint256 seed,
        uint8 domain,
        int8 sentiment
    ) internal view returns (string memory) {
        uint256[] memory domainTemps = domainTemplates[domain];
        if (domainTemps.length == 0) domainTemps = domainTemplates[DOMAIN_GENERAL];
        require(domainTemps.length > 0, "No templates available");

        uint256 templateIndex = domainTemps[seed % domainTemps.length];
        SentenceTemplate storage template = templates[templateIndex];
        if (!template.active) {
            for (uint256 i = 0; i < domainTemps.length; i++) {
                templateIndex = domainTemps[(seed + i) % domainTemps.length];
                template = templates[templateIndex];
                if (template.active) break;
            }
        }
        require(template.active, "No active templates found");

        string memory result = template.template;
        for (uint8 i = 0; i < template.slotCount; i++) {
            uint256 slotSeed = uint256(keccak256(abi.encode(seed, i)));
            uint256 wordId = getWordForSlot(slotSeed, template.slotTypes[i], domain, sentiment);
            string memory placeholder = string(abi.encodePacked("{", uint256(i).toString(), "}"));
            result = _replace(result, placeholder, vocabulary[wordId]);
        }
        return result;
    }

    function generateFromTransitions(
        uint256 seed,
        uint8 domain,
        int8 sentiment,
        uint256 maxLength,
        uint8 maxLayers
    ) internal view returns (string memory) {
        uint256 wordId = getStartWord(seed, domain, sentiment);
        require(wordId < vocabSize, "Failed to find start word");

        string memory result = vocabulary[wordId];
        uint256 length = 1;
        uint256 prevWordId = wordId;
        uint256[] memory context = new uint256[](maxLength);
        context[0] = wordId;

        while (length < maxLength) {
            uint256 nextSeed = uint256(keccak256(abi.encode(seed, length)));
            uint256 nextWordId = getNextWordWithLayers(nextSeed, prevWordId, wordId, context, length, maxLayers);

            if (nextWordId == vocabSize || keccak256(bytes(vocabulary[nextWordId])) == keccak256(bytes("."))) {
                break;
            }

            result = string(abi.encodePacked(result, " ", vocabulary[nextWordId]));
            prevWordId = wordId;
            wordId = nextWordId;
            context[length] = wordId;
            length++;
        }

        if (!_endsWith(result, ".") && !_endsWith(result, "!") && !_endsWith(result, "?")) {
            result = string(abi.encodePacked(result, "."));
        }
        return result;
    }

    function getWordForSlot(
        uint256 seed,
        uint8 slotType,
        uint8 domain,
        int8 sentiment
    ) internal view returns (uint256) {
        uint256[] memory candidates = new uint256[](100); // Limit candidates to avoid gas issues
        uint256[] memory scores = new uint256[](100);
        uint256 candidateCount = 0;

        // Collect candidate words
        for (uint256 i = 0; i < vocabSize && candidateCount < 100; i++) {
            WordMetadata memory meta = wordMetadata[i];

            // Check slot type constraint
            if (slotType != 0 && meta.partOfSpeech != slotType) continue;

            // Check domain relevance
            bool domainMatch = meta.domain == domain || meta.domain == DOMAIN_GENERAL;
            if (!domainMatch) continue;

            // Check sentiment alignment (within ±2 of target)
            int8 wordSentiment = int8(meta.sentiment) - 5;
            bool sentimentMatch = (wordSentiment >= sentiment - 2) && (wordSentiment <= sentiment + 2);
            if (!sentimentMatch) continue;

            // Add as candidate with initial score based on commonality
            candidates[candidateCount] = i;
            scores[candidateCount] = meta.commonality * 100; // Scale commonality
            candidateCount++;
        }

        // If no candidates, relax constraints
        if (candidateCount == 0) {
            for (uint256 i = 0; i < vocabSize && candidateCount < 100; i++) {
                if (slotType != 0 && wordMetadata[i].partOfSpeech != slotType) continue;
                candidates[candidateCount] = i;
                scores[candidateCount] = wordMetadata[i].commonality * 100;
                candidateCount++;
            }
        }

        // If still no candidates, return a random word
        if (candidateCount == 0) {
            return seed % vocabSize;
        }

        // Apply attention-based scoring
        for (uint256 i = 0; i < candidateCount; i++) {
            AttentionWeight[] storage weights = attentionWeights[candidates[i]];
            for (uint256 j = 0; j < weights.length; j++) {
                scores[i] += weights[j].weight; // Add attention weight
            }

            // Add positional weight
            PositionMetadata[] storage posMeta = wordPositionMetadata[candidates[i]];
            for (uint256 j = 0; j < posMeta.length; j++) {
                scores[i] += posMeta[j].weight / 2; // Decay positional influence
            }
        }

        // Select the highest-scoring candidate
        uint256 bestScore = 0;
        uint256 bestCandidate = candidates[0];
        for (uint256 i = 0; i < candidateCount; i++) {
            if (scores[i] > bestScore) {
                bestScore = scores[i];
                bestCandidate = candidates[i];
            }
        }

        return bestCandidate;
    }

    function getStartWord(
        uint256 seed,
        uint8 domain,
        int8 sentiment
    ) internal view returns (uint256) {
        // Try domain-specific start words
        uint256[] memory domainWords = domainStartWords[domain];
        if (domainWords.length > 0) {
            return domainWords[seed % domainWords.length];
        }

        // Try sentiment-specific start words
        uint256[] memory sentWords = sentimentStartWords[sentiment];
        if (sentWords.length > 0) {
            return sentWords[seed % sentWords.length];
        }

        // Fallback to random word
        return seed % vocabSize;
    }

    function getNextWordWithLayers(
        uint256 seed,
        uint256 prevWordId,
        uint256 wordId,
        uint256[] memory context,
        uint256 contextLength,
        uint8 maxLayers
    ) internal view returns (uint256) {
        uint256 finalWordId = vocabSize;
        uint256 highestScore = 0;

        // Iterate through layers
        for (uint8 layer = 0; layer < maxLayers && layer < MAX_LAYERS; layer++) {
            // Try bigram transitions
            LayeredTransition[] storage bigrams = layeredBigramTransitions[layer][prevWordId][wordId];
            if (bigrams.length > 0) {
                uint256 candidateId = selectFromTransitions(seed, bigrams);
                uint256 score = calculateAttentionScore(candidateId, context, contextLength, layer);
                if (score > highestScore) {
                    finalWordId = candidateId;
                    highestScore = score;
                }
            }

            // Try single word transitions
            LayeredTransition[] storage transitions = layeredWordTransitions[layer][wordId];
            if (transitions.length > 0) {
                uint256 candidateId = selectFromTransitions(seed, transitions);
                uint256 score = calculateAttentionScore(candidateId, context, contextLength, layer);
                if (score > highestScore) {
                    finalWordId = candidateId;
                    highestScore = score;
                }
            }
        }

        // Fallback to random word if no transitions found
        if (finalWordId == vocabSize) {
            return seed % vocabSize;
        }

        return finalWordId;
    }

    function selectFromTransitions(
        uint256 seed,
        LayeredTransition[] storage transitions
    ) internal view returns (uint256) {
        if (transitions.length == 1) {
            return transitions[0].nextWordId;
        }

        uint256 randomValue = seed % SCALE_FACTOR;
        uint256 cumulativeProb = 0;

        for (uint256 i = 0; i < transitions.length; i++) {
            cumulativeProb += transitions[i].probability;
            if (randomValue < cumulativeProb) {
                return transitions[i].nextWordId;
            }
        }

        return transitions[transitions.length - 1].nextWordId;
    }

    function calculateAttentionScore(
        uint256 candidateId,
        uint256[] memory context,
        uint256 contextLength,
        uint8 layer
    ) internal view returns (uint256) {
        uint256 score = 0;
        AttentionWeight[] storage weights = attentionWeights[candidateId];

        // Sum attention weights for context words
        for (uint256 i = 0; i < contextLength; i++) {
            for (uint256 j = 0; j < weights.length; j++) {
                if (weights[j].contextWordId == context[i]) {
                    score += weights[j].weight;
                }
            }
        }

        // Add positional weight
        PositionMetadata[] storage posMeta = wordPositionMetadata[candidateId];
        for (uint256 i = 0; i < posMeta.length; i++) {
            if (posMeta[i].position < contextLength) {
                score += posMeta[i].weight / (layer + 1); // Decay with layer
            }
        }

        return score;
    }

    // Utility Functions
    function _replace(
        string memory source,
        string memory searchStr,
        string memory replaceStr
    ) internal pure returns (string memory) {
        bytes memory sourceBytes = bytes(source);
        bytes memory searchBytes = bytes(searchStr);
        bytes memory replaceBytes = bytes(replaceStr);

        uint256 count = 0;
        for (uint256 i = 0; i <= sourceBytes.length - searchBytes.length; i++) {
            bool found = true;
            for (uint256 j = 0; j < searchBytes.length; j++) {
                if (sourceBytes[i + j] != searchBytes[j]) {
                    found = false;
                    break;
                }
            }
            if (found) count++;
        }

        if (count == 0) return source;

        bytes memory resultBytes = new bytes(sourceBytes.length + count * (replaceBytes.length - searchBytes.length));
        uint256 resultPos = 0;
        uint256 lastPos = 0;

        for (uint256 i = 0; i <= sourceBytes.length - searchBytes.length;) {
            bool found = true;
            for (uint256 j = 0; j < searchBytes.length; j++) {
                if (sourceBytes[i + j] != searchBytes[j]) {
                    found = false;
                    break;
                }
            }

            if (found) {
                for (uint256 j = lastPos; j < i; j++) {
                    resultBytes[resultPos++] = sourceBytes[j];
                }
                for (uint256 j = 0; j < replaceBytes.length; j++) {
                    resultBytes[resultPos++] = replaceBytes[j];
                }
                lastPos = i + searchBytes.length;
                i += searchBytes.length;
            } else {
                i++;
            }
        }

        for (uint256 i = lastPos; i < sourceBytes.length; i++) {
            resultBytes[resultPos++] = sourceBytes[i];
        }

        return string(resultBytes);
    }

    function _endsWith(string memory str, string memory suffix) internal pure returns (bool) {
        bytes memory strBytes = bytes(str);
        bytes memory suffixBytes = bytes(suffix);

        if (strBytes.length < suffixBytes.length) return false;

        for (uint256 i = 0; i < suffixBytes.length; i++) {
            if (strBytes[strBytes.length - suffixBytes.length + i] != suffixBytes[i]) {
                return false;
            }
        }
        return true;
    }

    // Admin Functions
    function pause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
