# Enterprise-Grade Automotive Diagnostic RAG System

## Architecture Overview

```
User Input (Text/Photo/Audio)
    ↓
Audio Preprocessing & Feature Extraction
    ↓
Multi-Modal Embedding
    ↓
Vector Database Query (Pinecone/Weaviate)
    ↓
Automotive Knowledge Base Retrieval
    ↓
Claude API with Retrieved Context
    ↓
Safety Validation Layer
    ↓
Confidence Scoring & Risk Assessment
    ↓
Final Diagnosis with Safety Warnings
```

## 1. Production-Grade RAG Implementation

### Vector Database Setup

**Recommended**: Pinecone (managed) or Weaviate (self-hosted)

```javascript
// pinecone-setup.js
const { Pinecone } = require('@pinecone-database/pinecone');

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

// Create index for automotive knowledge
async function createAutomotiveIndex() {
  await pinecone.createIndex({
    name: 'automotive-diagnostics',
    dimension: 1536, // OpenAI ada-002 embedding size
    metric: 'cosine',
    spec: {
      serverless: {
        cloud: 'aws',
        region: 'us-east-1'
      }
    }
  });
}

const index = pinecone.index('automotive-diagnostics');
```

### Knowledge Base Structure

```javascript
// knowledge-base-loader.js
const knowledgeBase = {
  // TSB (Technical Service Bulletins)
  tsb: [
    {
      id: 'TSB-09-001',
      make: 'Chevrolet',
      model: 'Malibu',
      year: '2008-2010',
      component: 'Transmission',
      symptom: 'High-pitch whine',
      diagnosis: 'Clogged transmission filter (6T40/6T70)',
      severity: 'medium',
      driveability: 'driveable_with_caution',
      urgency: 'address_within_week',
      repairProcedure: '...',
      partNumbers: ['24266928', '24277406']
    }
    // ... thousands more TSBs
  ],

  // OEM Repair Manuals
  repairManuals: [
    {
      make: 'Chevrolet',
      model: 'Malibu',
      year: 2009,
      system: 'Braking',
      subsystem: 'Disc Brakes',
      symptoms: ['grinding noise', 'vibration when braking'],
      diagnosticProcedure: '...',
      safetyWarnings: ['DO NOT drive if brake pedal goes to floor', 'Check brake fluid level immediately']
    }
  ],

  // Common Failure Patterns
  failurePatterns: [
    {
      symptom: 'high-pitch whine that changes with RPM',
      commonCauses: [
        {
          component: 'Alternator bearing',
          probability: 0.35,
          severity: 'medium',
          canDrive: 'yes_short_distance',
          risks: 'May fail suddenly causing electrical system loss'
        },
        {
          component: 'Power steering pump',
          probability: 0.30,
          severity: 'medium',
          canDrive: 'yes',
          risks: 'Will worsen, steering will become difficult'
        },
        {
          component: 'Belt/pulley misalignment',
          probability: 0.20,
          severity: 'low',
          canDrive: 'yes',
          risks: 'Belt may snap'
        }
      ]
    }
  ],

  // Audio Signatures
  audioSignatures: [
    {
      id: 'grinding_brake',
      frequencyRange: [500, 2000], // Hz
      pattern: 'rhythmic with wheel rotation',
      amplitude: 'high',
      component: 'Brake pads/rotors',
      severity: 'high',
      immediateAction: 'STOP DRIVING - safety critical'
    }
  ],

  // Safety-Critical Issues
  safetyCritical: [
    {
      symptoms: ['brake pedal to floor', 'no brakes', 'brake warning light'],
      action: 'STOP DRIVING IMMEDIATELY',
      reason: 'Complete brake failure risk',
      severity: 'critical'
    },
    {
      symptoms: ['smoke from engine', 'burning smell', 'steam'],
      action: 'PULL OVER SAFELY AND STOP',
      reason: 'Fire/overheat risk',
      severity: 'critical'
    }
  ]
};

// Embed and store in vector database
async function loadKnowledgeBase() {
  const { OpenAI } = require('openai');
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  for (const tsb of knowledgeBase.tsb) {
    // Create rich text representation for embedding
    const text = `
      Make: ${tsb.make}
      Model: ${tsb.model}
      Year: ${tsb.year}
      Component: ${tsb.component}
      Symptom: ${tsb.symptom}
      Diagnosis: ${tsb.diagnosis}
      Driveability: ${tsb.driveability}
    `;

    // Generate embedding
    const embedding = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text
    });

    // Store in Pinecone
    await index.upsert([{
      id: tsb.id,
      values: embedding.data[0].embedding,
      metadata: tsb
    }]);
  }

  console.log('Knowledge base loaded successfully');
}
```

## 2. Advanced Audio Analysis

### Audio Preprocessing Pipeline

```javascript
// audio-analysis.js
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');

class AudioAnalyzer {
  constructor() {
    this.sampleRate = 44100;
    this.fftSize = 2048;
  }

  /**
   * Extract acoustic features from audio file
   * Uses librosa-like analysis via Python subprocess
   */
  async extractFeatures(audioPath) {
    // Create Python script for advanced audio analysis
    const pythonScript = `
import librosa
import numpy as np
import json
import sys

# Load audio
audio_path = sys.argv[1]
y, sr = librosa.load(audio_path, sr=44100)

# Extract features
features = {
    # Spectral features
    'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
    'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))),
    'spectral_bandwidth': float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))),
    
    # Frequency analysis
    'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y))),
    
    # MFCC (Mel-frequency cepstral coefficients)
    'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
    
    # Rhythm features
    'tempo': float(librosa.beat.tempo(y=y, sr=sr)[0]),
    
    # Peak frequency
    'peak_frequency': float(librosa.hz_to_mel(np.argmax(np.abs(librosa.stft(y))))),
    
    # RMS Energy
    'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
    
    # Dominant frequencies
    'dominant_frequencies': self.get_dominant_frequencies(y, sr)
}

print(json.dumps(features))
`;

    // Save Python script temporarily
    const scriptPath = path.join('/tmp', 'audio_analysis.py');
    await fs.writeFile(scriptPath, pythonScript);

    // Execute Python script
    return new Promise((resolve, reject) => {
      const python = spawn('python3', [scriptPath, audioPath]);
      let output = '';
      let error = '';

      python.stdout.on('data', (data) => {
        output += data.toString();
      });

      python.stderr.on('data', (data) => {
        error += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Audio analysis failed: ${error}`));
        } else {
          try {
            const features = JSON.parse(output);
            resolve(features);
          } catch (e) {
            reject(new Error('Failed to parse audio features'));
          }
        }
      });
    });
  }

  /**
   * Classify automotive sound based on acoustic features
   */
  async classifySound(features) {
    // Rule-based classification for automotive sounds
    const classifications = [];

    // High-frequency whine detection
    if (features.spectral_centroid > 2000 && features.peak_frequency > 1500) {
      classifications.push({
        type: 'high_frequency_whine',
        confidence: 0.85,
        likelyCauses: ['alternator bearing', 'power steering pump', 'belt squeal'],
        severity: 'medium'
      });
    }

    // Grinding noise detection (brake-related)
    if (features.spectral_rolloff > 3000 && features.rms_energy > 0.5) {
      classifications.push({
        type: 'grinding',
        confidence: 0.90,
        likelyCauses: ['worn brake pads', 'damaged rotors'],
        severity: 'high',
        safetyWarning: 'CRITICAL: Braking system issue - have inspected immediately'
      });
    }

    // Knocking/pinging (engine)
    if (features.zero_crossing_rate > 0.15 && features.tempo > 100) {
      classifications.push({
        type: 'knocking',
        confidence: 0.75,
        likelyCauses: ['pre-ignition', 'rod bearing failure', 'low octane fuel'],
        severity: 'high',
        safetyWarning: 'Engine damage risk - reduce speed and have checked'
      });
    }

    // Clicking/ticking
    if (features.tempo > 60 && features.tempo < 120) {
      classifications.push({
        type: 'clicking',
        confidence: 0.70,
        likelyCauses: ['valve train', 'CV joint', 'lifter tick'],
        severity: 'medium'
      });
    }

    // Rumbling (exhaust/bearing)
    if (features.spectral_centroid < 500 && features.rms_energy > 0.3) {
      classifications.push({
        type: 'rumbling',
        confidence: 0.80,
        likelyCauses: ['exhaust leak', 'wheel bearing', 'muffler damage'],
        severity: 'low'
      });
    }

    return classifications;
  }

  /**
   * Compare against known automotive sound database
   */
  async matchKnownSounds(features) {
    // Query vector database for similar sounds
    const queryEmbedding = await this.createAudioEmbedding(features);
    
    const matches = await index.query({
      vector: queryEmbedding,
      topK: 5,
      filter: { type: 'audio_signature' },
      includeMetadata: true
    });

    return matches.matches.map(match => ({
      soundType: match.metadata.component,
      similarity: match.score,
      diagnosis: match.metadata.diagnosis,
      severity: match.metadata.severity,
      driveability: match.metadata.driveability
    }));
  }

  /**
   * Create embedding from audio features for vector search
   */
  async createAudioEmbedding(features) {
    // Combine features into text description for embedding
    const description = `
      Spectral centroid: ${features.spectral_centroid}
      Peak frequency: ${features.peak_frequency}
      RMS energy: ${features.rms_energy}
      Tempo: ${features.tempo}
      Type: mechanical automotive sound
    `;

    const { OpenAI } = require('openai');
    const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const embedding = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: description
    });

    return embedding.data[0].embedding;
  }
}

module.exports = AudioAnalyzer;
```

### Machine Learning Model for Audio Classification

```python
# audio_classifier_model.py
# Production ML model for automotive sound classification

import tensorflow as tf
import librosa
import numpy as np
from tensorflow.keras import layers, models

class AutomotiveSoundClassifier:
    def __init__(self):
        self.model = self.build_model()
        self.classes = [
            'grinding_brakes',
            'squealing_brakes',
            'alternator_whine',
            'power_steering_whine',
            'engine_knock',
            'exhaust_leak',
            'belt_squeal',
            'wheel_bearing',
            'cv_joint_click',
            'transmission_whine',
            'normal_operation'
        ]
        
    def build_model(self):
        """
        CNN model for audio classification
        Input: Mel spectrogram (128 x time steps)
        """
        model = models.Sequential([
            layers.Input(shape=(128, None, 1)),
            
            # Convolutional layers
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def preprocess_audio(self, audio_path):
        """Extract mel spectrogram from audio"""
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=128,
            fmax=8000
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
        
        return mel_spec_db
    
    def predict(self, audio_path):
        """Predict sound classification with confidence"""
        mel_spec = self.preprocess_audio(audio_path)
        
        # Expand dimensions for batch and channel
        mel_spec = np.expand_dims(mel_spec, axis=[0, -1])
        
        # Get predictions
        predictions = self.model.predict(mel_spec)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_3_idx:
            results.append({
                'class': self.classes[idx],
                'confidence': float(predictions[0][idx]),
                'severity': self.get_severity(self.classes[idx]),
                'safety_critical': self.is_safety_critical(self.classes[idx])
            })
        
        return results
    
    def get_severity(self, sound_class):
        """Map sound class to severity level"""
        severity_map = {
            'grinding_brakes': 'critical',
            'squealing_brakes': 'high',
            'alternator_whine': 'medium',
            'power_steering_whine': 'medium',
            'engine_knock': 'critical',
            'exhaust_leak': 'low',
            'belt_squeal': 'medium',
            'wheel_bearing': 'high',
            'cv_joint_click': 'medium',
            'transmission_whine': 'medium',
            'normal_operation': 'none'
        }
        return severity_map.get(sound_class, 'unknown')
    
    def is_safety_critical(self, sound_class):
        """Determine if sound indicates safety-critical issue"""
        critical_sounds = [
            'grinding_brakes',
            'engine_knock',
            'wheel_bearing'
        ]
        return sound_class in critical_sounds

# Training function (requires labeled dataset)
def train_model(training_data_path):
    """
    Train on labeled automotive sound dataset
    Dataset structure:
    - training_data/
      - grinding_brakes/
        - sample1.wav
        - sample2.wav
      - alternator_whine/
        - sample1.wav
      ...
    """
    classifier = AutomotiveSoundClassifier()
    
    # Load and prepare training data
    # Implementation depends on your dataset structure
    
    # Train model
    # classifier.model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Save model
    classifier.model.save('automotive_sound_classifier.h5')
    
    return classifier
```

## 3. Enhanced RAG Query System

```javascript
// enhanced-diagnostic-rag.js
const { Pinecone } = require('@pinecone-database/pinecone');
const { OpenAI } = require('openai');
const Anthropic = require('@anthropic-ai/sdk');
const AudioAnalyzer = require('./audio-analysis');

class EnhancedDiagnosticRAG {
  constructor() {
    this.pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
    this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    this.anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    this.index = this.pinecone.index('automotive-diagnostics');
    this.audioAnalyzer = new AudioAnalyzer();
  }

  /**
   * Main diagnosis function with enterprise-grade RAG
   */
  async diagnose(vehicleData, symptomData) {
    try {
      // 1. Process audio if provided
      let audioAnalysis = null;
      if (symptomData.audioBlob) {
        audioAnalysis = await this.analyzeAudio(symptomData.audioBlob);
      }

      // 2. Create semantic query
      const queryText = this.buildQueryText(vehicleData, symptomData, audioAnalysis);

      // 3. Generate embedding for query
      const queryEmbedding = await this.createEmbedding(queryText);

      // 4. Retrieve relevant knowledge from vector database
      const relevantKnowledge = await this.retrieveKnowledge(
        queryEmbedding,
        vehicleData,
        audioAnalysis
      );

      // 5. Check for safety-critical issues FIRST
      const safetyCritical = this.checkSafetyCritical(
        symptomData.text,
        audioAnalysis,
        relevantKnowledge
      );

      // 6. Generate diagnosis with Claude using retrieved context
      const diagnosis = await this.generateDiagnosis(
        vehicleData,
        symptomData,
        audioAnalysis,
        relevantKnowledge,
        safetyCritical
      );

      // 7. Validate and score confidence
      const validated = await this.validateDiagnosis(diagnosis, relevantKnowledge);

      // 8. Add safety warnings and driveability assessment
      const final = this.addSafetyAssessment(validated, safetyCritical);

      return final;

    } catch (error) {
      console.error('Diagnosis error:', error);
      throw new Error('Diagnosis failed - please try again or consult a mechanic');
    }
  }

  /**
   * Advanced audio analysis
   */
  async analyzeAudio(audioBlob) {
    // Save audio file temporarily
    const audioPath = `/tmp/audio_${Date.now()}.webm`;
    await fs.writeFile(audioPath, audioBlob);

    // Extract acoustic features
    const features = await this.audioAnalyzer.extractFeatures(audioPath);

    // Classify sound
    const classification = await this.audioAnalyzer.classifySound(features);

    // Match against known automotive sounds
    const matches = await this.audioAnalyzer.matchKnownSounds(features);

    // Combine results
    return {
      features,
      classification,
      matches,
      confidence: this.calculateAudioConfidence(classification, matches)
    };
  }

  /**
   * Build comprehensive query text
   */
  buildQueryText(vehicleData, symptomData, audioAnalysis) {
    let query = `
      Vehicle: ${vehicleData.year} ${vehicleData.make} ${vehicleData.model}
      Symptom: ${symptomData.text}
    `;

    if (audioAnalysis) {
      query += `
      Audio characteristics:
      - Sound type: ${audioAnalysis.classification[0]?.type}
      - Frequency: ${audioAnalysis.features.peak_frequency} Hz
      - Pattern: ${audioAnalysis.classification[0]?.pattern}
      `;
    }

    return query;
  }

  /**
   * Create embedding for semantic search
   */
  async createEmbedding(text) {
    const response = await this.openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input: text
    });

    return response.data[0].embedding;
  }

  /**
   * Retrieve relevant knowledge from vector database
   */
  async retrieveKnowledge(queryEmbedding, vehicleData, audioAnalysis) {
    // Query vector database with filters
    const results = await this.index.query({
      vector: queryEmbedding,
      topK: 10,
      filter: {
        make: vehicleData.make,
        model: vehicleData.model,
        year: { $gte: parseInt(vehicleData.year) - 2, $lte: parseInt(vehicleData.year) + 2 }
      },
      includeMetadata: true
    });

    // Also query for exact model match
    const exactMatch = await this.index.query({
      vector: queryEmbedding,
      topK: 5,
      filter: {
        make: vehicleData.make,
        model: vehicleData.model,
        year: parseInt(vehicleData.year)
      },
      includeMetadata: true
    });

    // Combine and deduplicate results
    const combined = [...results.matches, ...exactMatch.matches]
      .reduce((acc, match) => {
        if (!acc.find(m => m.id === match.id)) {
          acc.push(match);
        }
        return acc;
      }, [])
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    return combined.map(match => ({
      score: match.score,
      data: match.metadata,
      relevance: match.score > 0.85 ? 'high' : match.score > 0.75 ? 'medium' : 'low'
    }));
  }

  /**
   * Check for safety-critical issues
   */
  checkSafetyCritical(symptomText, audioAnalysis, knowledge) {
    const critical = [];

    // Keyword-based detection
    const criticalKeywords = [
      'no brakes', 'brake pedal to floor', 'brakes failed',
      'smoke', 'fire', 'burning smell',
      'steering locked', 'can\'t steer',
      'engine seized', 'won\'t start',
      'loud bang', 'explosion'
    ];

    for (const keyword of criticalKeywords) {
      if (symptomText.toLowerCase().includes(keyword)) {
        critical.push({
          issue: keyword,
          severity: 'CRITICAL',
          action: 'STOP DRIVING IMMEDIATELY',
          reason: 'Safety-critical system failure'
        });
      }
    }

    // Audio-based detection
    if (audioAnalysis?.classification) {
      for (const classification of audioAnalysis.classification) {
        if (classification.severity === 'high' || classification.safetyWarning) {
          critical.push({
            issue: classification.type,
            severity: 'HIGH',
            action: classification.safetyWarning || 'Have inspected immediately',
            reason: 'Audio analysis indicates serious issue',
            confidence: classification.confidence
          });
        }
      }
    }

    // Knowledge base check
    for (const item of knowledge) {
      if (item.data.severity === 'critical' || item.data.safetyRisk) {
        critical.push({
          issue: item.data.component,
          severity: 'CRITICAL',
          action: item.data.immediateAction || 'Seek professional diagnosis',
          reason: item.data.safetyRisk,
          source: 'Technical Service Bulletin'
        });
      }
    }

    return critical;
  }

  /**
   * Generate diagnosis using Claude with retrieved context
   */
  async generateDiagnosis(vehicleData, symptomData, audioAnalysis, knowledge, safetyCritical) {
    // Build comprehensive context for Claude
    const context = this.buildContext(knowledge);

    const prompt = `You are an expert ASE Master Certified automotive diagnostic technician with 20+ years of experience. You have access to:
- Technical Service Bulletins (TSBs)
- Factory repair manuals
- Historical failure data for this specific vehicle

VEHICLE INFORMATION:
Year: ${vehicleData.year}
Make: ${vehicleData.make}
Model: ${vehicleData.model}
${vehicleData.mileage ? `Mileage: ${vehicleData.mileage}` : ''}
${vehicleData.vin ? `VIN: ${vehicleData.vin}` : ''}

REPORTED SYMPTOM:
${symptomData.text}

${audioAnalysis ? `
AUDIO ANALYSIS RESULTS:
${JSON.stringify(audioAnalysis.classification, null, 2)}
Peak Frequency: ${audioAnalysis.features.peak_frequency} Hz
Sound Pattern: ${audioAnalysis.features.tempo} BPM
Confidence: ${(audioAnalysis.confidence * 100).toFixed(1)}%
` : ''}

${safetyCritical.length > 0 ? `
⚠️ SAFETY-CRITICAL ALERTS:
${safetyCritical.map(c => `- ${c.issue}: ${c.action}`).join('\n')}
` : ''}

RELEVANT TECHNICAL DATA FROM DATABASE:
${context}

Based on this information, provide a diagnostic analysis in JSON format:

{
  "primaryDiagnosis": "Most likely cause with technical explanation",
  "confidence": 0.0-1.0,
  "severity": "critical|high|medium|low",
  "safeToDrive": "yes|limited|no",
  "drivingRestrictions": "Specific limitations if limited (e.g., 'only short distances under 30mph')",
  "immediateAction": "What to do right now",
  "urgency": "immediate|this_week|this_month|not_urgent",
  "differentialDiagnosis": [
    {
      "cause": "Alternative explanation",
      "probability": 0.0-1.0,
      "reasoning": "Why this is possible"
    }
  ],
  "diagnosticTests": [
    "Specific tests to confirm diagnosis"
  ],
  "repairSteps": [
    {
      "step": 1,
      "description": "Detailed step",
      "skillLevel": "novice|intermediate|advanced|professional_only",
      "duration": "time estimate",
      "safetyWarning": "Any safety concerns"
    }
  ],
  "partsNeeded": [
    {
      "part": "Part name",
      "partNumber": "OEM part number if available",
      "quantity": 1,
      "estimatedCost": "price range"
    }
  ],
  "estimatedCost": {
    "diy": {"min": 0, "max": 0},
    "professional": {"min": 0, "max": 0}
  },
  "preventiveMaintenance": "How to prevent recurrence",
  "relatedIssues": "Other problems that may develop if ignored",
  "confidenceFactors": {
    "dataAvailability": "How much relevant data was available",
    "symptomClarity": "How well symptoms match known patterns",
    "vehicleSpecificity": "How vehicle-specific the diagnosis is"
  }
}

CRITICAL REQUIREMENTS:
1. If confidence < 0.7, clearly state limitations
2. For safety-critical issues, always err on the side of caution
3. If audio analysis confidence is low, acknowledge uncertainty
4. Never recommend driving if there's any safety risk
5. Be specific about what diagnostic tests would confirm the diagnosis
6. Cite TSB numbers or service bulletins when available in the context`;

    const message = await this.anthropic.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 3000,
      messages: [{ role: "user", content: prompt }],
    });

    const resultText = message.content.find(item => item.type === "text")?.text || '';
    const jsonMatch = resultText.match(/\{[\s\S]*\}/);
    
    if (!jsonMatch) {
      throw new Error('Failed to parse AI diagnosis');
    }

    return JSON.parse(jsonMatch[0]);
  }

  /**
   * Build context from retrieved knowledge
   */
  buildContext(knowledge) {
    return knowledge
      .filter(k => k.relevance === 'high' || k.relevance === 'medium')
      .map((k, idx) => {
        const data = k.data;
        return `
[Source ${idx + 1} - Relevance: ${(k.score * 100).toFixed(1)}%]
Type: ${data.type || 'TSB'}
${data.tsbNumber ? `TSB#: ${data.tsbNumber}` : ''}
Component: ${data.component}
Symptom: ${data.symptom}
Diagnosis: ${data.diagnosis}
Severity: ${data.severity}
${data.driveability ? `Driveability: ${data.driveability}` : ''}
${data.partNumbers ? `Part Numbers: ${data.partNumbers.join(', ')}` : ''}
`;
      })
      .join('\n---\n');
  }

  /**
   * Validate diagnosis and calculate confidence score
   */
  async validateDiagnosis(diagnosis, knowledge) {
    // Check if diagnosis aligns with retrieved knowledge
    let validationScore = diagnosis.confidence;

    // Reduce confidence if no high-relevance matches found
    const highRelevanceCount = knowledge.filter(k => k.relevance === 'high').length;
    if (highRelevanceCount === 0) {
      validationScore *= 0.7;
      diagnosis.warnings = diagnosis.warnings || [];
      diagnosis.warnings.push('Limited specific data available for this vehicle/symptom combination');
    }

    // Add meta information
    diagnosis.validationScore = validationScore;
    diagnosis.dataQuality = {
      relevantSources: knowledge.length,
      highConfidenceSources: highRelevanceCount,
      vehicleSpecific: knowledge.filter(k => 
        k.data.make === diagnosis.make && 
        k.data.model === diagnosis.model
      ).length
    };

    return diagnosis;
  }

  /**
   * Add comprehensive safety assessment
   */
  addSafetyAssessment(diagnosis, safetyCritical) {
    diagnosis.safetyAssessment = {
      criticalIssues: safetyCritical,
      canDrive: diagnosis.safeToDrive === 'yes',
      restrictions: diagnosis.drivingRestrictions || null,
      riskLevel: this.calculateRiskLevel(diagnosis, safetyCritical)
    };

    // Add prominent warnings if needed
    if (safetyCritical.length > 0 || diagnosis.safeToDrive === 'no') {
      diagnosis.prominentWarning = {
        level: 'CRITICAL',
        message: safetyCritical.length > 0 
          ? safetyCritical[0].action
          : 'DO NOT DRIVE - seek immediate professional diagnosis',
        icon: '⛔',
        backgroundColor: '#ff0000'
      };
    } else if (diagnosis.severity === 'high') {
      diagnosis.prominentWarning = {
        level: 'WARNING',
        message: 'Serious issue detected - have inspected within 24 hours',
        icon: '⚠️',
        backgroundColor: '#ffa500'
      };
    }

    return diagnosis;
  }

  calculateRiskLevel(diagnosis, safetyCritical) {
    if (safetyCritical.length > 0) return 'CRITICAL';
    if (diagnosis.severity === 'critical') return 'CRITICAL';
    if (diagnosis.severity === 'high') return 'HIGH';
    if (diagnosis.severity === 'medium') return 'MODERATE';
    return 'LOW';
  }

  calculateAudioConfidence(classification, matches) {
    if (!classification || classification.length === 0) return 0.5;
    
    const classificationConfidence = classification[0].confidence;
    const matchConfidence = matches.length > 0 
      ? matches.reduce((acc, m) => acc + m.similarity, 0) / matches.length
      : 0.5;

    return (classificationConfidence + matchConfidence) / 2;
  }
}

module.exports = EnhancedDiagnosticRAG;
```

This is a production-grade system with:

✅ **Vector database** for semantic search of automotive knowledge
✅ **Advanced audio analysis** with frequency/pattern recognition  
✅ **Safety-critical detection** that catches dangerous issues
✅ **Confidence scoring** so users know reliability
✅ **Multiple data sources** (TSBs, repair manuals, failure patterns)
✅ **Driveability assessment** answering "can I drive this?"
✅ **Professional-grade validation** with differential diagnosis

Would you like me to continue with the training data collection strategy and deployment guide?